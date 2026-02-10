export type ChatResponse = {
    title: string;
    create_time: number;
    update_time: number;
    mapping: MessageMapping;
    moderation_results: any[];
    current_node: string;
    plugin_ids: null;
    conversation_id: string;
    conversation_template_id: null;
    gizmo_id: null;
    gizmo_type: null;
    is_archived: boolean;
    is_starred: null;
    safe_urls: any[];
    default_model_slug: string;
    conversation_origin: null;
    voice: null;
    async_status: null;
    disabled_tool_ids: any[];
}

export type MessageMapping = {
    [key: string]: ChatNode;
}

export interface ChatNode {
    id: string;
    message: ChatMessage | null;
    parent: string | null;
    children: string[];
}

export interface ChatMessage {
    id: string;
    author: Author;
    create_time: number | null;
    update_time: null;
    content: MessageContent;
    status: string;
    end_turn: boolean | null;
    weight: number;
    metadata: BaseMetadata;
    recipient: string;
    channel: null;
}

export interface Author {
    role: string;
    name: string | null;
    metadata: Record<string, unknown>;
}

export type MessageContent = {
    content_type: string;
    parts?: string[];
    model_set_context?: string;
    repository?: null;
    repo_summary?: null;
    structured_context?: null;
}

interface BaseMetadata {
    canvas?: Canvas;
    command?: string;
    message_type: null;
    model_slug?: string;
    default_model_slug?: string;
    parent_id?: string;
    request_id?: string;
    timestamp_?: string;
    finish_details?: FinishDetails;
    is_complete?: boolean;
    citations?: any[];
    content_references?: any[];
    model_switcher_deny?: ModelSwitcherDeny[];
    is_visually_hidden_from_conversation?: boolean;
    serialization_metadata?: SerializationMetadata;
    message_source?: null;
}

interface Canvas {
    textdoc_id: string;
    textdoc_type: string;
    version: number;
    title: string;
    create_source: string;
}

interface FinishDetails {
    type: string;
    stop_tokens: number[];
}

interface ModelSwitcherDeny {
    slug: string | 'gpt-4o-mini' | 'o1-mini' | 'o3-mini';
    context: 'conversation' | 'regenerate';
    reason: 'unsupported_canvas';
    description: string;
}

interface SerializationMetadata {
    custom_symbol_offsets: any[];
}
