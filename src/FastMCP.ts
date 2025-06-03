/**
 * FastMCP - 高性能 Model Context Protocol (MCP) 实现
 *
 * @fileoverview
 * FastMCP 是一个用于构建 MCP 服务器的 TypeScript 框架。MCP 是一种标准化协议，
 * 用于在 AI 模型（如 Claude）和外部工具/数据源之间建立通信桥梁。
 *
 * 核心架构：
 * - FastMCP: 主服务器类，管理多个会话和服务器生命周期
 * - FastMCPSession: 单个客户端会话，处理具体的协议通信
 * - 三大功能系统：Tools（工具）、Resources（资源）、Prompts（提示）
 *
 * 传输方式：
 * - stdio: 标准输入/输出（适用于本地进程间通信）
 * - httpStream: HTTP 流式传输（适用于网络通信）
 *
 * @module FastMCP
 */

// MCP SDK 核心组件 - 提供协议实现和传输层抽象
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
// MCP 协议模式定义 - 定义请求/响应的数据结构
import {
  CallToolRequestSchema,
  ClientCapabilities,
  CompleteRequestSchema,
  CreateMessageRequestSchema,
  ErrorCode,
  GetPromptRequestSchema,
  ListPromptsRequestSchema,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ReadResourceRequestSchema,
  Root,
  RootsListChangedNotificationSchema,
  ServerCapabilities,
  SetLevelRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
// 类型系统和验证
import { StandardSchemaV1 } from "@standard-schema/spec";
import { EventEmitter } from "events";
import { fileTypeFromBuffer } from "file-type";
import { readFile } from "fs/promises";
import Fuse from "fuse.js";
import http from "http";
import { startHTTPServer } from "mcp-proxy";
import { StrictEventEmitter } from "strict-event-emitter-types";
import { setTimeout as delay } from "timers/promises";
import { fetch } from "undici";
import parseURITemplate from "uri-templates";
import { toJsonSchema } from "xsschema";
import { z } from "zod";

/**
 * SSE（Server-Sent Events）服务器接口
 * 用于 HTTP 流式传输模式下的服务器管理
 */
export type SSEServer = {
  close: () => Promise<void>;
};

/**
 * FastMCP 服务器级别事件
 * @template T - 认证信息类型
 */
type FastMCPEvents<T extends FastMCPSessionAuth> = {
  /** 当新的客户端会话建立连接时触发 */
  connect: (event: { session: FastMCPSession<T> }) => void;
  /** 当客户端会话断开连接时触发 */
  disconnect: (event: { session: FastMCPSession<T> }) => void;
};

/**
 * FastMCP 会话级别事件
 */
type FastMCPSessionEvents = {
  /** 会话发生错误时触发 */
  error: (event: { error: Error }) => void;
  /** 根目录列表发生变化时触发（用于文件系统访问） */
  rootsChanged: (event: { roots: Root[] }) => void;
};

/**
 * 将图片数据转换为 MCP 协议所需的 ImageContent 格式
 *
 * @description
 * 支持三种输入方式：
 * - URL: 从网络获取图片
 * - 文件路径: 从本地文件系统读取
 * - Buffer: 直接处理内存中的图片数据
 *
 * 自动检测图片的 MIME 类型，并将图片数据编码为 base64 格式。
 *
 * @param input - 图片输入源
 * @returns Promise<ImageContent> - 包含 base64 编码数据和 MIME 类型的图片内容对象
 *
 * @throws {Error} 当无法获取图片或图片格式无效时
 *
 * @example
 * ```typescript
 * // 从 URL 加载
 * const urlImage = await imageContent({ url: 'https://example.com/image.png' });
 *
 * // 从文件加载
 * const fileImage = await imageContent({ path: './local-image.jpg' });
 *
 * // 从 Buffer 加载
 * const bufferImage = await imageContent({ buffer: imageBuffer });
 * ```
 */
export const imageContent = async (
  input: { buffer: Buffer } | { path: string } | { url: string }
): Promise<ImageContent> => {
  let rawData: Buffer;

  try {
    // 处理不同的输入源
    if ("url" in input) {
      try {
        const response = await fetch(input.url);

        if (!response.ok) {
          throw new Error(
            `Server responded with status: ${response.status} - ${response.statusText}`
          );
        }

        rawData = Buffer.from(await response.arrayBuffer());
      } catch (error) {
        throw new Error(
          `Failed to fetch image from URL (${input.url}): ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } else if ("path" in input) {
      try {
        rawData = await readFile(input.path);
      } catch (error) {
        throw new Error(
          `Failed to read image from path (${input.path}): ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } else if ("buffer" in input) {
      rawData = input.buffer;
    } else {
      throw new Error(
        "Invalid input: Provide a valid 'url', 'path', or 'buffer'"
      );
    }

    // 检测文件类型以确定 MIME 类型
    const mimeType = await fileTypeFromBuffer(rawData);

    if (!mimeType || !mimeType.mime.startsWith("image/")) {
      console.warn(
        `Warning: Content may not be a valid image. Detected MIME: ${mimeType?.mime || "unknown"}`
      );
    }

    // 将二进制数据编码为 base64
    const base64Data = rawData.toString("base64");

    return {
      data: base64Data,
      mimeType: mimeType?.mime ?? "image/png", // 默认使用 PNG 格式
      type: "image",
    } as const;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    } else {
      throw new Error(`Unexpected error processing image: ${String(error)}`);
    }
  }
};

/**
 * 将音频数据转换为 MCP 协议所需的 AudioContent 格式
 *
 * @description
 * 支持三种输入方式：
 * - URL: 从网络获取音频
 * - 文件路径: 从本地文件系统读取
 * - Buffer: 直接处理内存中的音频数据
 *
 * 自动检测音频的 MIME 类型，并将音频数据编码为 base64 格式。
 *
 * @param input - 音频输入源
 * @returns Promise<AudioContent> - 包含 base64 编码数据和 MIME 类型的音频内容对象
 *
 * @throws {Error} 当无法获取音频或音频格式无效时
 *
 * @example
 * ```typescript
 * // 从 URL 加载
 * const urlAudio = await audioContent({ url: 'https://example.com/audio.mp3' });
 *
 * // 从文件加载
 * const fileAudio = await audioContent({ path: './local-audio.wav' });
 *
 * // 从 Buffer 加载
 * const bufferAudio = await audioContent({ buffer: audioBuffer });
 * ```
 */
export const audioContent = async (
  input: { buffer: Buffer } | { path: string } | { url: string }
): Promise<AudioContent> => {
  let rawData: Buffer;

  try {
    if ("url" in input) {
      try {
        const response = await fetch(input.url);

        if (!response.ok) {
          throw new Error(
            `Server responded with status: ${response.status} - ${response.statusText}`
          );
        }

        rawData = Buffer.from(await response.arrayBuffer());
      } catch (error) {
        throw new Error(
          `Failed to fetch audio from URL (${input.url}): ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } else if ("path" in input) {
      try {
        rawData = await readFile(input.path);
      } catch (error) {
        throw new Error(
          `Failed to read audio from path (${input.path}): ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } else if ("buffer" in input) {
      rawData = input.buffer;
    } else {
      throw new Error(
        "Invalid input: Provide a valid 'url', 'path', or 'buffer'"
      );
    }

    const mimeType = await fileTypeFromBuffer(rawData);

    if (!mimeType || !mimeType.mime.startsWith("audio/")) {
      console.warn(
        `Warning: Content may not be a valid audio file. Detected MIME: ${mimeType?.mime || "unknown"}`
      );
    }

    const base64Data = rawData.toString("base64");

    return {
      data: base64Data,
      mimeType: mimeType?.mime ?? "audio/mpeg",
      type: "audio",
    } as const;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    } else {
      throw new Error(`Unexpected error processing audio: ${String(error)}`);
    }
  }
};

/**
 * 工具执行上下文
 *
 * @description
 * 提供给工具执行函数的上下文对象，包含日志记录、进度报告、会话信息和内容流式传输等功能。
 * 这是工具与 MCP 框架交互的主要接口。
 *
 * @template T - 会话认证信息的类型
 */
type Context<T extends FastMCPSessionAuth> = {
  /**
   * 日志记录器
   * 提供不同级别的日志输出功能，日志会发送到客户端
   */
  log: {
    /** 调试级别日志 */
    debug: (message: string, data?: SerializableValue) => void;
    /** 错误级别日志 */
    error: (message: string, data?: SerializableValue) => void;
    /** 信息级别日志 */
    info: (message: string, data?: SerializableValue) => void;
    /** 警告级别日志 */
    warn: (message: string, data?: SerializableValue) => void;
  };
  /**
   * 报告工具执行进度
   * 允许长时间运行的工具向客户端报告执行进度
   */
  reportProgress: (progress: Progress) => Promise<void>;
  /**
   * 当前会话的认证信息
   * 如果服务器配置了认证，这里会包含认证后的用户信息
   */
  session: T | undefined;
  /**
   * 流式传输内容
   * 允许工具在执行过程中逐步发送内容，而不必等到完全执行完毕
   */
  streamContent: (content: Content | Content[]) => Promise<void>;
};

type Extra = unknown;

type Extras = Record<string, Extra>;

type Literal = boolean | null | number | string | undefined;

/**
 * 进度信息
 *
 * @description
 * 用于报告工具或操作的执行进度。支持已知总量和未知总量两种场景。
 */
type Progress = {
  /**
   * 当前进度值
   * 应该在每次取得进展时递增，即使总量未知
   */
  progress: number;
  /**
   * 总进度量（如果已知）
   * 表示完成操作所需的总工作量
   */
  total?: number;
};

/**
 * 可序列化值类型
 *
 * @description
 * 定义可以安全序列化为 JSON 的值类型。
 * 用于日志记录和数据传输。
 */
type SerializableValue =
  | { [key: string]: SerializableValue }
  | Literal
  | SerializableValue[];

/**
 * 文本内容类型
 *
 * @description
 * MCP 协议中最基本的内容类型，用于传输纯文本信息。
 */
type TextContent = {
  /** 文本内容 */
  text: string;
  /** 内容类型标识符 */
  type: "text";
};

/**
 * 工具参数类型
 *
 * @description
 * 工具参数必须符合 StandardSchemaV1 规范，
 * 这确保了参数可以被验证和类型推断。
 */
type ToolParameters = StandardSchemaV1;

/**
 * FastMCP 错误基类
 *
 * @description
 * 所有 FastMCP 特定错误的抽象基类。
 * 自动设置错误名称为类名。
 */
abstract class FastMCPError extends Error {
  public constructor(message?: string) {
    super(message);
    this.name = new.target.name;
  }
}

/**
 * 意外状态错误
 *
 * @description
 * 表示系统遇到了意外的状态或条件。
 * 可以包含额外的上下文信息用于调试。
 */
export class UnexpectedStateError extends FastMCPError {
  /** 额外的错误上下文信息 */
  public extras?: Extras;

  public constructor(message: string, extras?: Extras) {
    super(message);
    this.name = new.target.name;
    this.extras = extras;
  }
}

/**
 * 用户错误
 *
 * @description
 * 表示由用户操作或输入导致的错误。
 * 这些错误的消息会直接显示给用户，因此应该是友好和可理解的。
 *
 * @extends UnexpectedStateError
 */
export class UserError extends UnexpectedStateError {}

const TextContentZodSchema = z
  .object({
    /**
     * The text content of the message.
     */
    text: z.string(),
    type: z.literal("text"),
  })
  .strict() satisfies z.ZodType<TextContent>;

/**
 * 图片内容类型
 *
 * @description
 * 用于在 MCP 协议中传输图片数据。
 * 图片数据以 base64 编码传输。
 */
type ImageContent = {
  /** Base64 编码的图片数据 */
  data: string;
  /** 图片的 MIME 类型（如 'image/png', 'image/jpeg'） */
  mimeType: string;
  /** 内容类型标识符 */
  type: "image";
};

const ImageContentZodSchema = z
  .object({
    /**
     * The base64-encoded image data.
     */
    data: z.string().base64(),
    /**
     * The MIME type of the image. Different providers may support different image types.
     */
    mimeType: z.string(),
    type: z.literal("image"),
  })
  .strict() satisfies z.ZodType<ImageContent>;

/**
 * 音频内容类型
 *
 * @description
 * 用于在 MCP 协议中传输音频数据。
 * 音频数据以 base64 编码传输。
 */
type AudioContent = {
  /** Base64 编码的音频数据 */
  data: string;
  /** 音频的 MIME 类型（如 'audio/mpeg', 'audio/wav'） */
  mimeType: string;
  /** 内容类型标识符 */
  type: "audio";
};

const AudioContentZodSchema = z
  .object({
    /**
     * The base64-encoded audio data.
     */
    data: z.string().base64(),
    mimeType: z.string(),
    type: z.literal("audio"),
  })
  .strict() satisfies z.ZodType<AudioContent>;

/**
 * 内容类型联合
 *
 * @description
 * MCP 支持的所有内容类型的联合类型。
 * 使用 type 字段进行区分。
 */
type Content = AudioContent | ImageContent | TextContent;

const ContentZodSchema = z.discriminatedUnion("type", [
  TextContentZodSchema,
  ImageContentZodSchema,
  AudioContentZodSchema,
]) satisfies z.ZodType<Content>;

/**
 * 内容结果类型
 *
 * @description
 * 工具执行的标准返回格式。
 * 可以包含多个内容项，并标记是否为错误结果。
 */
type ContentResult = {
  /** 内容数组 */
  content: Content[];
  /** 是否为错误结果 */
  isError?: boolean;
};

const ContentResultZodSchema = z
  .object({
    content: ContentZodSchema.array(),
    isError: z.boolean().optional(),
  })
  .strict() satisfies z.ZodType<ContentResult>;

/**
 * 自动完成结果类型
 *
 * @description
 * 用于参数自动完成功能的返回类型。
 * 支持分页和部分结果返回。
 */
type Completion = {
  /** 是否有更多结果未返回 */
  hasMore?: boolean;
  /** 可用选项的总数（可能超过实际返回的数量） */
  total?: number;
  /** 完成选项值数组（最多100项） */
  values: string[];
};

/**
 * https://github.com/modelcontextprotocol/typescript-sdk/blob/3164da64d085ec4e022ae881329eee7b72f208d4/src/types.ts#L983-L1003
 */
const CompletionZodSchema = z.object({
  /**
   * Indicates whether there are additional completion options beyond those provided in the current response, even if the exact total is unknown.
   */
  hasMore: z.optional(z.boolean()),
  /**
   * The total number of completion options available. This can exceed the number of values actually sent in the response.
   */
  total: z.optional(z.number().int()),
  /**
   * An array of completion values. Must not exceed 100 items.
   */
  values: z.array(z.string()).max(100),
}) satisfies z.ZodType<Completion>;

/**
 * 参数值自动完成函数类型
 *
 * @description
 * 用于提供参数值的自动完成建议。
 * 接收当前输入值，返回匹配的建议列表。
 */
type ArgumentValueCompleter = (value: string) => Promise<Completion>;

type InputPrompt<
  Arguments extends InputPromptArgument[] = InputPromptArgument[],
  Args = PromptArgumentsToObject<Arguments>,
> = {
  arguments?: InputPromptArgument[];
  description?: string;
  load: (args: Args) => Promise<string>;
  name: string;
};

type InputPromptArgument = Readonly<{
  complete?: ArgumentValueCompleter;
  description?: string;
  enum?: string[];
  name: string;
  required?: boolean;
}>;

type InputResourceTemplate<
  Arguments extends ResourceTemplateArgument[] = ResourceTemplateArgument[],
> = {
  arguments: Arguments;
  description?: string;
  load: (
    args: ResourceTemplateArgumentsToObject<Arguments>
  ) => Promise<ResourceResult>;
  mimeType?: string;
  name: string;
  uriTemplate: string;
};

type InputResourceTemplateArgument = Readonly<{
  complete?: ArgumentValueCompleter;
  description?: string;
  name: string;
  required?: boolean;
}>;

type LoggingLevel =
  | "alert"
  | "critical"
  | "debug"
  | "emergency"
  | "error"
  | "info"
  | "notice"
  | "warning";

type Prompt<
  Arguments extends PromptArgument[] = PromptArgument[],
  Args = PromptArgumentsToObject<Arguments>,
> = {
  arguments?: PromptArgument[];
  complete?: (name: string, value: string) => Promise<Completion>;
  description?: string;
  load: (args: Args) => Promise<string>;
  name: string;
};

type PromptArgument = Readonly<{
  complete?: ArgumentValueCompleter;
  description?: string;
  enum?: string[];
  name: string;
  required?: boolean;
}>;

type PromptArgumentsToObject<T extends { name: string; required?: boolean }[]> =
  {
    [K in T[number]["name"]]: Extract<
      T[number],
      { name: K }
    >["required"] extends true
      ? string
      : string | undefined;
  };

type Resource = {
  complete?: (name: string, value: string) => Promise<Completion>;
  description?: string;
  load: () => Promise<ResourceResult | ResourceResult[]>;
  mimeType?: string;
  name: string;
  uri: string;
};

/**
 * 资源结果类型
 *
 * @description
 * 资源加载的返回类型。
 * 支持文本和二进制（blob）两种格式。
 */
type ResourceResult =
  | {
      /** Base64 编码的二进制数据 */
      blob: string;
    }
  | {
      /** 文本数据 */
      text: string;
    };

type ResourceTemplate<
  Arguments extends ResourceTemplateArgument[] = ResourceTemplateArgument[],
> = {
  arguments: Arguments;
  complete?: (name: string, value: string) => Promise<Completion>;
  description?: string;
  load: (
    args: ResourceTemplateArgumentsToObject<Arguments>
  ) => Promise<ResourceResult>;
  mimeType?: string;
  name: string;
  uriTemplate: string;
};

type ResourceTemplateArgument = Readonly<{
  complete?: ArgumentValueCompleter;
  description?: string;
  name: string;
  required?: boolean;
}>;

type ResourceTemplateArgumentsToObject<T extends { name: string }[]> = {
  [K in T[number]["name"]]: string;
};

type ServerOptions<T extends FastMCPSessionAuth> = {
  authenticate?: Authenticate<T>;
  /**
   * Configuration for the health-check endpoint that can be exposed when the
   * server is running using the HTTP Stream transport. When enabled, the
   * server will respond to an HTTP GET request with the configured path (by
   * default "/health") rendering a plain-text response (by default "ok") and
   * the configured status code (by default 200).
   *
   * The endpoint is only added when the server is started with
   * `transportType: "httpStream"` – it is ignored for the stdio transport.
   */
  health?: {
    /**
     * When set to `false` the health-check endpoint is disabled.
     * @default true
     */
    enabled?: boolean;

    /**
     * Plain-text body returned by the endpoint.
     * @default "ok"
     */
    message?: string;

    /**
     * HTTP path that should be handled.
     * @default "/health"
     */
    path?: string;

    /**
     * HTTP response status that will be returned.
     * @default 200
     */
    status?: number;
  };
  instructions?: string;
  name: string;

  ping?: {
    /**
     * Whether ping should be enabled by default.
     * - true for SSE or HTTP Stream
     * - false for stdio
     */
    enabled?: boolean;
    /**
     * Interval
     * @default 5000 (5s)
     */
    intervalMs?: number;
    /**
     * Logging level for ping-related messages.
     * @default 'debug'
     */
    logLevel?: LoggingLevel;
  };
  /**
   * Configuration for roots capability
   */
  roots?: {
    /**
     * Whether roots capability should be enabled
     * Set to false to completely disable roots support
     * @default true
     */
    enabled?: boolean;
  };
  version: `${number}.${number}.${number}`;
};

/**
 * 工具定义
 *
 * @description
 * 工具是 MCP 的核心功能之一，允许 AI 模型调用外部功能。
 * 每个工具都有明确定义的参数模式和执行函数。
 *
 * @template T - 会话认证信息的类型
 * @template Params - 工具参数的模式类型，必须符合 StandardSchemaV1
 *
 * @example
 * ```typescript
 * const calculateTool: Tool<undefined> = {
 *   name: 'calculate',
 *   description: '执行数学计算',
 *   parameters: z.object({
 *     expression: z.string().describe('数学表达式')
 *   }),
 *   execute: async (args, context) => {
 *     context.log.info('计算表达式', { expression: args.expression });
 *     const result = eval(args.expression); // 示例，实际应使用安全的计算库
 *     return `结果: ${result}`;
 *   }
 * };
 * ```
 */
type Tool<
  T extends FastMCPSessionAuth,
  Params extends ToolParameters = ToolParameters,
> = {
  /**
   * 工具注解
   * 提供关于工具行为的额外元数据
   */
  annotations?: {
    /**
     * 指示工具是否使用增量内容流
     * 当为 true 时，工具会通过 streamContent 发送内容，execute 可返回 void
     */
    streamingHint?: boolean;
  } & ToolAnnotations;
  /** 工具的描述，帮助 AI 理解何时使用此工具 */
  description?: string;
  /**
   * 工具的执行函数
   * @param args - 经过验证的参数对象
   * @param context - 执行上下文
   * @returns 工具执行结果，可以是文本、图片、音频或复合内容
   */
  execute: (
    args: StandardSchemaV1.InferOutput<Params>,
    context: Context<T>
  ) => Promise<
    AudioContent | ContentResult | ImageContent | string | TextContent | void
  >;
  /** 工具的唯一名称 */
  name: string;
  /** 工具参数的模式定义，用于参数验证 */
  parameters?: Params;
  /** 工具执行的超时时间（毫秒） */
  timeoutMs?: number;
};

/**
 * 工具注解
 *
 * @description
 * 定义在 MCP 规范 (2025-03-26) 中的工具注解。
 * 这些注解提供关于工具行为的提示信息，帮助 AI 模型更好地理解和使用工具。
 */
type ToolAnnotations = {
  /**
   * 工具是否可能执行破坏性更新
   * 仅在 readOnlyHint 为 false 时有意义
   * @default true
   */
  destructiveHint?: boolean;

  /**
   * 使用相同参数重复调用工具是否无额外效果（幂等性）
   * 仅在 readOnlyHint 为 false 时有意义
   * @default false
   */
  idempotentHint?: boolean;

  /**
   * 工具是否可能与外部实体交互
   * @default true
   */
  openWorldHint?: boolean;

  /**
   * 指示工具不会修改其环境（只读）
   * @default false
   */
  readOnlyHint?: boolean;

  /**
   * 工具的人类可读标题
   * 用于 UI 显示
   */
  title?: string;
};

const FastMCPSessionEventEmitterBase: {
  new (): StrictEventEmitter<EventEmitter, FastMCPSessionEvents>;
} = EventEmitter;

/**
 * 会话认证信息类型
 *
 * @description
 * 会话可以包含任意的认证信息对象，或者为 undefined（无认证）。
 */
type FastMCPSessionAuth = Record<string, unknown> | undefined;

/**
 * 采样响应类型
 *
 * @description
 * AI 模型生成响应的格式。
 * 包含生成的内容、使用的模型信息和停止原因。
 */
type SamplingResponse = {
  /** 生成的内容 */
  content: AudioContent | ImageContent | TextContent;
  /** 使用的模型标识 */
  model: string;
  /** 响应角色 */
  role: "assistant" | "user";
  /** 生成停止的原因 */
  stopReason?: "endTurn" | "maxTokens" | "stopSequence" | string;
};

class FastMCPSessionEventEmitter extends FastMCPSessionEventEmitterBase {}

/**
 * FastMCP 会话类
 *
 * @description
 * 代表一个客户端与 MCP 服务器之间的会话连接。
 * 每个会话管理自己的：
 * - 工具、资源和提示的注册
 * - 与客户端的通信
 * - 能力协商
 * - 日志级别
 * - 认证信息
 *
 * 在 stdio 模式下，通常只有一个会话；
 * 在 httpStream 模式下，可以有多个并发会话。
 *
 * @template T - 会话认证信息的类型
 *
 * @extends FastMCPSessionEventEmitter
 */
export class FastMCPSession<
  T extends FastMCPSessionAuth = FastMCPSessionAuth,
> extends FastMCPSessionEventEmitter {
  public get clientCapabilities(): ClientCapabilities | null {
    return this.#clientCapabilities ?? null;
  }
  public get loggingLevel(): LoggingLevel {
    return this.#loggingLevel;
  }
  public get roots(): Root[] {
    return this.#roots;
  }
  public get server(): Server {
    return this.#server;
  }
  #auth: T | undefined;
  #capabilities: ServerCapabilities = {};
  #clientCapabilities?: ClientCapabilities;
  #loggingLevel: LoggingLevel = "info";
  #pingConfig?: ServerOptions<T>["ping"];
  #pingInterval: null | ReturnType<typeof setInterval> = null;

  #prompts: Prompt[] = [];

  #resources: Resource[] = [];

  #resourceTemplates: ResourceTemplate[] = [];

  #roots: Root[] = [];

  #rootsConfig?: ServerOptions<T>["roots"];

  #server: Server;

  /**
   * 创建新的 FastMCP 会话
   *
   * @param options - 会话配置选项
   * @param options.auth - 认证信息（如果有）
   * @param options.instructions - 服务器指令，提供给 AI 模型的上下文信息
   * @param options.name - 服务器名称
   * @param options.ping - ping 配置
   * @param options.prompts - 要注册的提示列表
   * @param options.resources - 要注册的资源列表
   * @param options.resourcesTemplates - 要注册的资源模板列表
   * @param options.roots - 根目录配置
   * @param options.tools - 要注册的工具列表
   * @param options.version - 服务器版本
   */
  constructor({
    auth,
    instructions,
    name,
    ping,
    prompts,
    resources,
    resourcesTemplates,
    roots,
    tools,
    version,
  }: {
    auth?: T;
    instructions?: string;
    name: string;
    ping?: ServerOptions<T>["ping"];
    prompts: Prompt[];
    resources: Resource[];
    resourcesTemplates: InputResourceTemplate[];
    roots?: ServerOptions<T>["roots"];
    tools: Tool<T>[];
    version: string;
  }) {
    super();

    this.#auth = auth;
    this.#pingConfig = ping;
    this.#rootsConfig = roots;

    // 根据注册的功能设置服务器能力
    if (tools.length) {
      this.#capabilities.tools = {};
    }

    if (resources.length || resourcesTemplates.length) {
      this.#capabilities.resources = {};
    }

    if (prompts.length) {
      for (const prompt of prompts) {
        this.addPrompt(prompt);
      }

      this.#capabilities.prompts = {};
    }

    this.#capabilities.logging = {};

    // 创建 MCP 服务器实例
    this.#server = new Server(
      { name: name, version: version },
      { capabilities: this.#capabilities, instructions: instructions }
    );

    // 设置各种处理器
    this.setupErrorHandling();
    this.setupLoggingHandlers();
    this.setupRootsHandlers();
    this.setupCompleteHandlers();

    if (tools.length) {
      this.setupToolHandlers(tools);
    }

    if (resources.length || resourcesTemplates.length) {
      for (const resource of resources) {
        this.addResource(resource);
      }

      this.setupResourceHandlers(resources);

      if (resourcesTemplates.length) {
        for (const resourceTemplate of resourcesTemplates) {
          this.addResourceTemplate(resourceTemplate);
        }

        this.setupResourceTemplateHandlers(resourcesTemplates);
      }
    }

    if (prompts.length) {
      this.setupPromptHandlers(prompts);
    }
  }

  /**
   * 关闭会话并清理资源
   *
   * @description
   * 停止 ping 定时器（如果有）并关闭底层服务器连接。
   * 即使关闭过程中出现错误，也会尽力完成清理工作。
   */
  public async close() {
    if (this.#pingInterval) {
      clearInterval(this.#pingInterval);
    }

    try {
      await this.#server.close();
    } catch (error) {
      console.error("[FastMCP error]", "could not close server", error);
    }
  }

  /**
   * 建立与客户端的连接
   *
   * @description
   * 连接流程：
   * 1. 建立传输层连接
   * 2. 获取客户端能力（最多尝试 10 次）
   * 3. 如果客户端支持，获取根目录列表
   * 4. 根据传输类型和配置启动 ping 机制
   *
   * @param transport - 传输层实现（stdio 或 httpStream）
   *
   * @throws {UnexpectedStateError} 如果服务器已经连接
   */
  public async connect(transport: Transport) {
    if (this.#server.transport) {
      throw new UnexpectedStateError("Server is already connected");
    }

    await this.#server.connect(transport);

    // 等待客户端能力信息
    let attempt = 0;

    while (attempt++ < 10) {
      const capabilities = await this.#server.getClientCapabilities();

      if (capabilities) {
        this.#clientCapabilities = capabilities;

        break;
      }

      await delay(100);
    }

    if (!this.#clientCapabilities) {
      console.warn("[FastMCP warning] could not infer client capabilities");
    }

    // 如果客户端支持根目录变更通知，尝试获取根目录列表
    if (
      this.#clientCapabilities?.roots?.listChanged &&
      typeof this.#server.listRoots === "function"
    ) {
      try {
        const roots = await this.#server.listRoots();
        this.#roots = roots.roots;
      } catch (e) {
        if (e instanceof McpError && e.code === ErrorCode.MethodNotFound) {
          console.debug(
            "[FastMCP debug] listRoots method not supported by client"
          );
        } else {
          console.error(
            `[FastMCP error] received error listing roots.\n\n${e instanceof Error ? e.stack : JSON.stringify(e)}`
          );
        }
      }
    }

    // 配置并启动 ping 机制
    if (this.#clientCapabilities) {
      const pingConfig = this.#getPingConfig(transport);

      if (pingConfig.enabled) {
        this.#pingInterval = setInterval(async () => {
          try {
            await this.#server.ping();
          } catch {
            // 某些客户端可能不响应 ping 请求，我们不希望因此崩溃
            // 参见：https://github.com/punkpeye/fastmcp/issues/38
            const logLevel = pingConfig.logLevel;
            if (logLevel === "debug") {
              console.debug("[FastMCP debug] server ping failed");
            } else if (logLevel === "warning") {
              console.warn(
                "[FastMCP warning] server is not responding to ping"
              );
            } else if (logLevel === "error") {
              console.error("[FastMCP error] server is not responding to ping");
            } else {
              console.info("[FastMCP info] server ping failed");
            }
          }
        }, pingConfig.intervalMs);
      }
    }
  }

  public async requestSampling(
    message: z.infer<typeof CreateMessageRequestSchema>["params"]
  ): Promise<SamplingResponse> {
    return this.#server.createMessage(message);
  }

  #getPingConfig(transport: Transport): {
    enabled: boolean;
    intervalMs: number;
    logLevel: LoggingLevel;
  } {
    const pingConfig = this.#pingConfig || {};

    let defaultEnabled = false;

    if ("type" in transport) {
      // Enable by default for SSE and HTTP streaming
      if (transport.type === "httpStream") {
        defaultEnabled = true;
      }
    }

    return {
      enabled:
        pingConfig.enabled !== undefined ? pingConfig.enabled : defaultEnabled,
      intervalMs: pingConfig.intervalMs || 5000,
      logLevel: pingConfig.logLevel || "debug",
    };
  }

  private addPrompt(inputPrompt: InputPrompt) {
    const completers: Record<string, ArgumentValueCompleter> = {};
    const enums: Record<string, string[]> = {};

    for (const argument of inputPrompt.arguments ?? []) {
      if (argument.complete) {
        completers[argument.name] = argument.complete;
      }

      if (argument.enum) {
        enums[argument.name] = argument.enum;
      }
    }

    const prompt = {
      ...inputPrompt,
      complete: async (name: string, value: string) => {
        if (completers[name]) {
          return await completers[name](value);
        }

        if (enums[name]) {
          const fuse = new Fuse(enums[name], {
            keys: ["value"],
          });

          const result = fuse.search(value);

          return {
            total: result.length,
            values: result.map((item) => item.item),
          };
        }

        return {
          values: [],
        };
      },
    };

    this.#prompts.push(prompt);
  }

  private addResource(inputResource: Resource) {
    this.#resources.push(inputResource);
  }

  private addResourceTemplate(inputResourceTemplate: InputResourceTemplate) {
    const completers: Record<string, ArgumentValueCompleter> = {};

    for (const argument of inputResourceTemplate.arguments ?? []) {
      if (argument.complete) {
        completers[argument.name] = argument.complete;
      }
    }

    const resourceTemplate = {
      ...inputResourceTemplate,
      complete: async (name: string, value: string) => {
        if (completers[name]) {
          return await completers[name](value);
        }

        return {
          values: [],
        };
      },
    };

    this.#resourceTemplates.push(resourceTemplate);
  }

  private setupCompleteHandlers() {
    this.#server.setRequestHandler(CompleteRequestSchema, async (request) => {
      if (request.params.ref.type === "ref/prompt") {
        const prompt = this.#prompts.find(
          (prompt) => prompt.name === request.params.ref.name
        );

        if (!prompt) {
          throw new UnexpectedStateError("Unknown prompt", {
            request,
          });
        }

        if (!prompt.complete) {
          throw new UnexpectedStateError("Prompt does not support completion", {
            request,
          });
        }

        const completion = CompletionZodSchema.parse(
          await prompt.complete(
            request.params.argument.name,
            request.params.argument.value
          )
        );

        return {
          completion,
        };
      }

      if (request.params.ref.type === "ref/resource") {
        const resource = this.#resourceTemplates.find(
          (resource) => resource.uriTemplate === request.params.ref.uri
        );

        if (!resource) {
          throw new UnexpectedStateError("Unknown resource", {
            request,
          });
        }

        if (!("uriTemplate" in resource)) {
          throw new UnexpectedStateError("Unexpected resource");
        }

        if (!resource.complete) {
          throw new UnexpectedStateError(
            "Resource does not support completion",
            {
              request,
            }
          );
        }

        const completion = CompletionZodSchema.parse(
          await resource.complete(
            request.params.argument.name,
            request.params.argument.value
          )
        );

        return {
          completion,
        };
      }

      throw new UnexpectedStateError("Unexpected completion request", {
        request,
      });
    });
  }

  private setupErrorHandling() {
    this.#server.onerror = (error) => {
      console.error("[FastMCP error]", error);
    };
  }

  private setupLoggingHandlers() {
    this.#server.setRequestHandler(SetLevelRequestSchema, (request) => {
      this.#loggingLevel = request.params.level;

      return {};
    });
  }

  private setupPromptHandlers(prompts: Prompt[]) {
    this.#server.setRequestHandler(ListPromptsRequestSchema, async () => {
      return {
        prompts: prompts.map((prompt) => {
          return {
            arguments: prompt.arguments,
            complete: prompt.complete,
            description: prompt.description,
            name: prompt.name,
          };
        }),
      };
    });

    this.#server.setRequestHandler(GetPromptRequestSchema, async (request) => {
      const prompt = prompts.find(
        (prompt) => prompt.name === request.params.name
      );

      if (!prompt) {
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown prompt: ${request.params.name}`
        );
      }

      const args = request.params.arguments;

      for (const arg of prompt.arguments ?? []) {
        if (arg.required && !(args && arg.name in args)) {
          throw new McpError(
            ErrorCode.InvalidRequest,
            `Prompt '${request.params.name}' requires argument '${arg.name}': ${arg.description || "No description provided"}`
          );
        }
      }

      let result: Awaited<ReturnType<Prompt["load"]>>;

      try {
        result = await prompt.load(args as Record<string, string | undefined>);
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        throw new McpError(
          ErrorCode.InternalError,
          `Failed to load prompt '${request.params.name}': ${errorMessage}`
        );
      }

      return {
        description: prompt.description,
        messages: [
          {
            content: { text: result, type: "text" },
            role: "user",
          },
        ],
      };
    });
  }

  private setupResourceHandlers(resources: Resource[]) {
    this.#server.setRequestHandler(ListResourcesRequestSchema, async () => {
      return {
        resources: resources.map((resource) => {
          return {
            mimeType: resource.mimeType,
            name: resource.name,
            uri: resource.uri,
          };
        }),
      };
    });

    this.#server.setRequestHandler(
      ReadResourceRequestSchema,
      async (request) => {
        if ("uri" in request.params) {
          const resource = resources.find(
            (resource) =>
              "uri" in resource && resource.uri === request.params.uri
          );

          if (!resource) {
            for (const resourceTemplate of this.#resourceTemplates) {
              const uriTemplate = parseURITemplate(
                resourceTemplate.uriTemplate
              );

              const match = uriTemplate.fromUri(request.params.uri);

              if (!match) {
                continue;
              }

              const uri = uriTemplate.fill(match);

              const result = await resourceTemplate.load(match);

              return {
                contents: [
                  {
                    mimeType: resourceTemplate.mimeType,
                    name: resourceTemplate.name,
                    uri: uri,
                    ...result,
                  },
                ],
              };
            }

            throw new McpError(
              ErrorCode.MethodNotFound,
              `Resource not found: '${request.params.uri}'. Available resources: ${resources.map((r) => r.uri).join(", ") || "none"}`
            );
          }

          if (!("uri" in resource)) {
            throw new UnexpectedStateError("Resource does not support reading");
          }

          let maybeArrayResult: Awaited<ReturnType<Resource["load"]>>;

          try {
            maybeArrayResult = await resource.load();
          } catch (error) {
            const errorMessage =
              error instanceof Error ? error.message : String(error);
            throw new McpError(
              ErrorCode.InternalError,
              `Failed to load resource '${resource.name}' (${resource.uri}): ${errorMessage}`,
              {
                uri: resource.uri,
              }
            );
          }

          if (Array.isArray(maybeArrayResult)) {
            return {
              contents: maybeArrayResult.map((result) => ({
                mimeType: resource.mimeType,
                name: resource.name,
                uri: resource.uri,
                ...result,
              })),
            };
          } else {
            return {
              contents: [
                {
                  mimeType: resource.mimeType,
                  name: resource.name,
                  uri: resource.uri,
                  ...maybeArrayResult,
                },
              ],
            };
          }
        }

        throw new UnexpectedStateError("Unknown resource request", {
          request,
        });
      }
    );
  }

  private setupResourceTemplateHandlers(resourceTemplates: ResourceTemplate[]) {
    this.#server.setRequestHandler(
      ListResourceTemplatesRequestSchema,
      async () => {
        return {
          resourceTemplates: resourceTemplates.map((resourceTemplate) => {
            return {
              name: resourceTemplate.name,
              uriTemplate: resourceTemplate.uriTemplate,
            };
          }),
        };
      }
    );
  }

  private setupRootsHandlers() {
    if (this.#rootsConfig?.enabled === false) {
      console.debug(
        "[FastMCP debug] roots capability explicitly disabled via config"
      );
      return;
    }

    // Only set up roots notification handling if the server supports it
    if (typeof this.#server.listRoots === "function") {
      this.#server.setNotificationHandler(
        RootsListChangedNotificationSchema,
        () => {
          this.#server
            .listRoots()
            .then((roots) => {
              this.#roots = roots.roots;

              this.emit("rootsChanged", {
                roots: roots.roots,
              });
            })
            .catch((error) => {
              if (
                error instanceof McpError &&
                error.code === ErrorCode.MethodNotFound
              ) {
                console.debug(
                  "[FastMCP debug] listRoots method not supported by client"
                );
              } else {
                console.error("[FastMCP error] Error listing roots", error);
              }
            });
        }
      );
    } else {
      console.debug(
        "[FastMCP debug] roots capability not available, not setting up notification handler"
      );
    }
  }

  /**
   * 设置工具处理器
   *
   * @description
   * 注册工具相关的请求处理器：
   * - ListToolsRequestSchema: 返回所有可用工具的列表和模式
   * - CallToolRequestSchema: 执行具体的工具调用
   *
   * 工具执行流程：
   * 1. 验证工具是否存在
   * 2. 验证和解析参数
   * 3. 创建执行上下文（日志、进度报告、内容流等）
   * 4. 执行工具函数（支持超时）
   * 5. 处理和格式化返回结果
   *
   * @param tools - 要注册的工具列表
   * @private
   */
  private setupToolHandlers(tools: Tool<T>[]) {
    // 处理列出工具的请求
    this.#server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: await Promise.all(
          tools.map(async (tool) => {
            return {
              annotations: tool.annotations,
              description: tool.description,
              inputSchema: tool.parameters
                ? await toJsonSchema(tool.parameters)
                : {
                    additionalProperties: false,
                    properties: {},
                    type: "object",
                  }, // 为 Cursor 兼容性提供更完整的模式
              name: tool.name,
            };
          })
        ),
      };
    });

    // 处理调用工具的请求
    this.#server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const tool = tools.find((tool) => tool.name === request.params.name);

      if (!tool) {
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown tool: ${request.params.name}`
        );
      }

      let args: unknown = undefined;

      // 验证和解析参数
      if (tool.parameters) {
        const parsed = await tool.parameters["~standard"].validate(
          request.params.arguments
        );

        if (parsed.issues) {
          const friendlyErrors = parsed.issues
            .map((issue) => {
              const path = issue.path?.join(".") || "root";
              return `${path}: ${issue.message}`;
            })
            .join(", ");

          throw new McpError(
            ErrorCode.InvalidParams,
            `Tool '${request.params.name}' parameter validation failed: ${friendlyErrors}`
          );
        }

        args = parsed.value;
      }

      const progressToken = request.params?._meta?.progressToken;

      let result: ContentResult;

      try {
        // 创建进度报告函数
        const reportProgress = async (progress: Progress) => {
          await this.#server.notification({
            method: "notifications/progress",
            params: {
              ...progress,
              progressToken,
            },
          });
        };

        // 创建日志记录函数
        const log = {
          debug: (message: string, context?: SerializableValue) => {
            this.#server.sendLoggingMessage({
              data: {
                context,
                message,
              },
              level: "debug",
            });
          },
          error: (message: string, context?: SerializableValue) => {
            this.#server.sendLoggingMessage({
              data: {
                context,
                message,
              },
              level: "error",
            });
          },
          info: (message: string, context?: SerializableValue) => {
            this.#server.sendLoggingMessage({
              data: {
                context,
                message,
              },
              level: "info",
            });
          },
          warn: (message: string, context?: SerializableValue) => {
            this.#server.sendLoggingMessage({
              data: {
                context,
                message,
              },
              level: "warning",
            });
          },
        };

        // 创建内容流传输函数
        // 在工具仍在执行时流式传输部分结果
        // 实现渐进式渲染和实时反馈
        const streamContent = async (content: Content | Content[]) => {
          const contentArray = Array.isArray(content) ? content : [content];

          await this.#server.notification({
            method: "notifications/tool/streamContent",
            params: {
              content: contentArray,
              toolName: request.params.name,
            },
          });
        };

        // 执行工具
        const executeToolPromise = tool.execute(args, {
          log,
          reportProgress,
          session: this.#auth,
          streamContent,
        });

        // 处理超时（如果指定了超时时间）
        const maybeStringResult = (await (tool.timeoutMs
          ? Promise.race([
              executeToolPromise,
              new Promise<never>((_, reject) => {
                setTimeout(() => {
                  reject(
                    new UserError(
                      `Tool '${request.params.name}' timed out after ${tool.timeoutMs}ms. Consider increasing timeoutMs or optimizing the tool implementation.`
                    )
                  );
                }, tool.timeoutMs);
              }),
            ])
          : executeToolPromise)) as
          | AudioContent
          | ContentResult
          | ImageContent
          | null
          | string
          | TextContent
          | undefined;

        // 格式化返回结果
        if (maybeStringResult === undefined || maybeStringResult === null) {
          result = ContentResultZodSchema.parse({
            content: [],
          });
        } else if (typeof maybeStringResult === "string") {
          result = ContentResultZodSchema.parse({
            content: [{ text: maybeStringResult, type: "text" }],
          });
        } else if ("type" in maybeStringResult) {
          result = ContentResultZodSchema.parse({
            content: [maybeStringResult],
          });
        } else {
          result = ContentResultZodSchema.parse(maybeStringResult);
        }
      } catch (error) {
        // 用户错误（预期的错误）显示给用户
        if (error instanceof UserError) {
          return {
            content: [{ text: error.message, type: "text" }],
            isError: true,
          };
        }

        // 其他错误作为工具执行失败处理
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        return {
          content: [
            {
              text: `Tool '${request.params.name}' execution failed: ${errorMessage}`,
              type: "text",
            },
          ],
          isError: true,
        };
      }

      return result;
    });
  }
}

const FastMCPEventEmitterBase: {
  new (): StrictEventEmitter<EventEmitter, FastMCPEvents<FastMCPSessionAuth>>;
} = EventEmitter;

/**
 * HTTP 认证函数类型
 *
 * @description
 * 用于 httpStream 模式下的请求认证。
 * 接收 HTTP 请求，返回认证信息。
 *
 * @template T - 认证信息的类型
 */
type Authenticate<T> = (request: http.IncomingMessage) => Promise<T>;

class FastMCPEventEmitter extends FastMCPEventEmitterBase {}

/**
 * FastMCP 主服务器类
 *
 * @description
 * FastMCP 是整个框架的入口点，负责：
 * - 管理服务器生命周期（启动、停止）
 * - 管理多个客户端会话
 * - 注册全局的工具、资源和提示
 * - 处理不同的传输模式（stdio、httpStream）
 * - 提供认证机制（仅适用于 httpStream）
 *
 * 使用示例：
 * ```typescript
 * const server = new FastMCP({
 *   name: 'my-mcp-server',
 *   version: '1.0.0',
 *   instructions: '这是一个示例 MCP 服务器'
 * });
 *
 * server.addTool({
 *   name: 'hello',
 *   description: '打招呼',
 *   execute: async () => 'Hello, World!'
 * });
 *
 * await server.start(); // 默认使用 stdio
 * ```
 *
 * @template T - 认证信息的类型，默认为 undefined（无认证）
 *
 * @extends FastMCPEventEmitter
 */
export class FastMCP<
  T extends Record<string, unknown> | undefined = undefined,
> extends FastMCPEventEmitter {
  public get sessions(): FastMCPSession<T>[] {
    return this.#sessions;
  }
  #authenticate: Authenticate<T> | undefined;
  #httpStreamServer: null | SSEServer = null;
  #options: ServerOptions<T>;
  #prompts: InputPrompt[] = [];
  #resources: Resource[] = [];
  #resourcesTemplates: InputResourceTemplate[] = [];
  #sessions: FastMCPSession<T>[] = [];

  #tools: Tool<T>[] = [];

  constructor(public options: ServerOptions<T>) {
    super();

    this.#options = options;
    this.#authenticate = options.authenticate;
  }

  /**
   * 添加提示到服务器
   *
   * @description
   * 提示会在所有新建立的会话中可用
   */
  public addPrompt<const Args extends InputPromptArgument[]>(
    prompt: InputPrompt<Args>
  ) {
    this.#prompts.push(prompt);
  }

  /**
   * 添加资源到服务器
   *
   * @description
   * 资源会在所有新建立的会话中可用
   */
  public addResource(resource: Resource) {
    this.#resources.push(resource);
  }

  /**
   * 添加资源模板到服务器
   *
   * @description
   * 资源模板允许使用 URI 模板动态生成资源
   */
  public addResourceTemplate<
    const Args extends InputResourceTemplateArgument[],
  >(resource: InputResourceTemplate<Args>) {
    this.#resourcesTemplates.push(resource);
  }

  /**
   * 添加工具到服务器
   *
   * @description
   * 工具会在所有新建立的会话中可用
   */
  public addTool<Params extends ToolParameters>(tool: Tool<T, Params>) {
    this.#tools.push(tool as unknown as Tool<T>);
  }

  /**
   * 启动 MCP 服务器
   *
   * @description
   * 支持两种传输模式：
   * - stdio（默认）：通过标准输入输出通信，适用于本地进程
   * - httpStream：通过 HTTP 流通信，适用于网络环境
   *
   * stdio 模式：
   * - 立即创建单个会话
   * - 适用于命令行工具和本地集成
   *
   * httpStream 模式：
   * - 监听指定端口
   * - 为每个连接创建新会话
   * - 支持认证
   * - 提供健康检查端点
   *
   * @param options - 启动配置
   * @param options.transportType - 传输类型
   * @param options.httpStream - HTTP 流配置（仅在 transportType 为 'httpStream' 时需要）
   * @param options.httpStream.port - 监听端口
   *
   * @example
   * ```typescript
   * // stdio 模式（默认）
   * await server.start();
   *
   * // httpStream 模式
   * await server.start({
   *   transportType: 'httpStream',
   *   httpStream: { port: 3000 }
   * });
   * ```
   */
  public async start(
    options:
      | {
          httpStream: { port: number };
          transportType: "httpStream";
        }
      | { transportType: "stdio" } = {
      transportType: "stdio",
    }
  ) {
    if (options.transportType === "stdio") {
      // stdio 模式：创建单个会话
      const transport = new StdioServerTransport();

      const session = new FastMCPSession<T>({
        instructions: this.#options.instructions,
        name: this.#options.name,
        ping: this.#options.ping,
        prompts: this.#prompts,
        resources: this.#resources,
        resourcesTemplates: this.#resourcesTemplates,
        roots: this.#options.roots,
        tools: this.#tools,
        version: this.#options.version,
      });

      await session.connect(transport);

      this.#sessions.push(session);

      this.emit("connect", {
        session,
      });
    } else if (options.transportType === "httpStream") {
      // httpStream 模式：启动 HTTP 服务器
      this.#httpStreamServer = await startHTTPServer<FastMCPSession<T>>({
        createServer: async (request) => {
          let auth: T | undefined;

          // 执行认证（如果配置了认证函数）
          if (this.#authenticate) {
            auth = await this.#authenticate(request);
          }

          return new FastMCPSession<T>({
            auth,
            name: this.#options.name,
            ping: this.#options.ping,
            prompts: this.#prompts,
            resources: this.#resources,
            resourcesTemplates: this.#resourcesTemplates,
            roots: this.#options.roots,
            tools: this.#tools,
            version: this.#options.version,
          });
        },
        onClose: (session) => {
          this.emit("disconnect", {
            session,
          });
        },
        onConnect: async (session) => {
          this.#sessions.push(session);

          this.emit("connect", {
            session,
          });
        },
        onUnhandledRequest: async (req, res) => {
          const healthConfig = this.#options.health ?? {};

          const enabled =
            healthConfig.enabled === undefined ? true : healthConfig.enabled;

          if (enabled) {
            const path = healthConfig.path ?? "/health";

            try {
              if (
                req.method === "GET" &&
                new URL(req.url || "", "http://localhost").pathname === path
              ) {
                res
                  .writeHead(healthConfig.status ?? 200, {
                    "Content-Type": "text/plain",
                  })
                  .end(healthConfig.message ?? "ok");

                return;
              }
            } catch (error) {
              console.error("[FastMCP error] health endpoint error", error);
            }
          }

          // 未处理的请求返回 404
          res.writeHead(404).end();
        },
        port: options.httpStream.port,
      });

      console.info(
        `[FastMCP info] server is running on HTTP Stream at http://localhost:${options.httpStream.port}/stream`
      );
    } else {
      throw new Error("Invalid transport type");
    }
  }

  /**
   * 停止 MCP 服务器
   *
   * @description
   * 关闭 HTTP 服务器（如果在 httpStream 模式下运行）。
   * stdio 模式下会自动在进程退出时清理。
   */
  public async stop() {
    if (this.#httpStreamServer) {
      await this.#httpStreamServer.close();
    }
  }
}

// 导出类型定义，供外部使用
/** 工具执行上下文类型 */
export type { Context };
/** 工具定义和参数类型 */
export type { Tool, ToolParameters };
/** 内容类型定义 */
export type { AudioContent, Content, ContentResult, ImageContent, TextContent };
/** 进度和序列化值类型 */
export type { Progress, SerializableValue };
/** 资源相关类型 */
export type { Resource, ResourceResult };
/** 资源模板相关类型 */
export type { ResourceTemplate, ResourceTemplateArgument };
/** 提示相关类型 */
export type { Prompt, PromptArgument };
/** 输入提示相关类型 */
export type { InputPrompt, InputPromptArgument };
/** 日志级别和服务器选项 */
export type { LoggingLevel, ServerOptions };
/** 事件类型 */
export type { FastMCPEvents, FastMCPSessionEvents };
