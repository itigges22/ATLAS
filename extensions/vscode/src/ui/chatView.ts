// Chat sidebar: a WebviewViewProvider that owns all turn state and treats
// the webview as a dumb renderer. Every message posted to the webview is
// also appended to a transcript so a re-created webview (sidebar closed and
// reopened, window reload of the view) can be replayed from scratch.

import * as vscode from 'vscode';
import { AtlasApiError, AtlasClient } from '../client/atlasClient';
import type {
	ErrorEventData,
	PermissionMode,
	PermissionRequestEventData,
	TextEventData,
	ToolCallEventData,
	ToolResultEventData,
} from '../client/types';
import { TurnManager } from '../session/turnManager';

/** SecretStorage key for the service token ('ATLAS: Set Service Token'). */
export const TOKEN_SECRET_KEY = 'atlas.serviceToken';

/** Messages the extension posts INTO the webview (media/chat.js). */
type OutboundMessage =
	| { type: 'userMessage'; text: string }
	| { type: 'assistantDelta'; text: string }
	| { type: 'toolCall'; name: string; detail: string }
	| { type: 'toolResult'; tool: string; success: boolean; elapsed?: string; error?: string }
	| { type: 'note'; text: string }
	| { type: 'turnDone' }
	| { type: 'turnError'; message: string }
	| { type: 'reset' }
	| { type: 'busy'; value: boolean };

/** Messages the webview posts back to the extension. */
type InboundMessage = { type: 'ready' } | { type: 'submit'; text: string } | { type: 'cancel' };

/** Condense tool args to a single short line for the tool chip. */
function condenseArgs(args: unknown): string {
	if (args === null || args === undefined) {
		return '';
	}
	let text: string;
	try {
		text = typeof args === 'string' ? args : JSON.stringify(args);
	} catch {
		return '';
	}
	return text.length > 120 ? `${text.slice(0, 117)}...` : text;
}

export class ChatViewProvider implements vscode.WebviewViewProvider {
	static readonly viewType = 'atlas.chatView';

	private view: vscode.WebviewView | undefined;
	/** Replayed into any freshly created webview; `busy` is transient and
	 * excluded. */
	private transcript: OutboundMessage[] = [];
	private readonly turns = new TurnManager();
	private readonly output: vscode.OutputChannel;

	constructor(
		private readonly extensionUri: vscode.Uri,
		private readonly secrets: vscode.SecretStorage,
	) {
		this.output = vscode.window.createOutputChannel('ATLAS');
	}

	resolveWebviewView(view: vscode.WebviewView): void {
		this.view = view;
		view.webview.options = {
			enableScripts: true,
			localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, 'media')],
		};
		view.webview.html = this.renderHtml(view.webview);
		view.webview.onDidReceiveMessage((message: InboundMessage) => {
			switch (message.type) {
				case 'ready':
					this.replay();
					break;
				case 'submit':
					void this.runTurn(message.text);
					break;
				case 'cancel':
					this.cancelTurn();
					break;
			}
		});
	}

	cancelTurn(): void {
		this.turns.cancel();
	}

	newConversation(): void {
		this.turns.cancel();
		this.turns.reset();
		this.transcript = [];
		this.postTransient({ type: 'reset' });
	}

	private async runTurn(text: string): Promise<void> {
		const message = text.trim();
		if (message === '') {
			return;
		}
		if (this.turns.busy) {
			this.post({ type: 'note', text: 'A turn is already in progress — cancel it first.' });
			return;
		}

		const config = vscode.workspace.getConfiguration('atlas');
		const baseUrl = config.get<string>('proxyUrl', 'http://localhost:8090');
		const mode = config.get<PermissionMode>('permissionMode', 'default');
		// Plaintext setting is a dev override; SecretStorage is the real home.
		const token = config.get<string>('serviceToken', '') || (await this.secrets.get(TOKEN_SECRET_KEY)) || '';
		const client = new AtlasClient({ baseUrl, token });

		this.post({ type: 'userMessage', text: message });
		this.postTransient({ type: 'busy', value: true });
		try {
			for await (const event of this.turns.runTurn(client, message, mode)) {
				this.dispatch(event.type, event.data, client);
			}
			this.post({ type: 'turnDone' });
		} catch (error) {
			this.handleTurnFailure(error);
		} finally {
			this.postTransient({ type: 'busy', value: false });
		}
	}

	private dispatch(type: string, data: unknown, client: AtlasClient): void {
		switch (type) {
			case 'text': {
				const payload = data as TextEventData;
				if (typeof payload?.content === 'string') {
					this.post({ type: 'assistantDelta', text: payload.content });
				}
				break;
			}
			case 'tool_call': {
				const payload = data as ToolCallEventData;
				this.post({ type: 'toolCall', name: payload.name, detail: condenseArgs(payload.args) });
				break;
			}
			case 'tool_result': {
				const payload = data as ToolResultEventData;
				this.post({
					type: 'toolResult',
					tool: payload.tool,
					success: payload.success,
					elapsed: payload.elapsed,
					error: payload.error,
				});
				break;
			}
			case 'permission_request': {
				// Interim commit-3 behavior: deny immediately and say so, so
				// the turn never sits on the 600s server-side timeout. The
				// real approve/deny flow lands in the next commit.
				const payload = data as PermissionRequestEventData;
				this.post({
					type: 'note',
					text: `Permission request for '${payload.tool_name}' auto-denied — the approval UI lands in an upcoming commit.`,
				});
				void client
					.postPermissionDecision({
						session_id: this.turns.sessionId,
						tool_call_id: payload.tool_call_id,
						decision: 'deny',
						scope: 'once',
					})
					.catch((error: unknown) => this.log('permission deny failed', error));
				break;
			}
			case 'error': {
				const payload = data as ErrorEventData;
				this.post({ type: 'turnError', message: payload.error || 'unknown stream error' });
				break;
			}
			case 'done':
				// Bubble finalization happens on generator completion.
				break;
			default:
				// Forward compatibility: unknown event types are logged, never fatal.
				this.log(`unhandled event '${type}'`, data);
		}
	}

	private handleTurnFailure(error: unknown): void {
		if (error instanceof Error && error.name === 'AbortError') {
			this.post({ type: 'note', text: 'Turn cancelled.' });
			return;
		}
		if (error instanceof AtlasApiError) {
			this.post({ type: 'turnError', message: `${error.code || 'request failed'}: ${error.detail || error.message}` });
			if (error.code === 'unauthorized') {
				void vscode.window
					.showErrorMessage('ATLAS proxy rejected the request (unauthorized).', 'Set Token')
					.then((choice) => {
						if (choice === 'Set Token') {
							void vscode.commands.executeCommand('atlas.setToken');
						}
					});
			}
			return;
		}
		const message = error instanceof Error ? error.message : String(error);
		this.post({ type: 'turnError', message: `Could not reach the ATLAS proxy: ${message}` });
	}

	/** Post to the webview and record for replay. */
	private post(message: OutboundMessage): void {
		this.transcript.push(message);
		void this.view?.webview.postMessage(message);
	}

	/** Post without recording (busy toggles, reset marker). */
	private postTransient(message: OutboundMessage): void {
		void this.view?.webview.postMessage(message);
	}

	private replay(): void {
		for (const message of this.transcript) {
			void this.view?.webview.postMessage(message);
		}
		this.postTransient({ type: 'busy', value: this.turns.busy });
	}

	private log(context: string, detail: unknown): void {
		let rendered: string;
		try {
			rendered = JSON.stringify(detail);
		} catch {
			rendered = String(detail);
		}
		this.output.appendLine(`${context}: ${rendered}`);
	}

	private renderHtml(webview: vscode.Webview): string {
		const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.js'));
		const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.css'));
		const nonce = getNonce();
		return `<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta http-equiv="Content-Security-Policy"
		content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<link href="${styleUri}" rel="stylesheet">
	<title>ATLAS Chat</title>
</head>
<body>
	<main id="messages" aria-live="polite"></main>
	<footer id="composer">
		<textarea id="input" rows="2" placeholder="Ask ATLAS..."></textarea>
		<div id="actions">
			<button id="send" type="button">Send</button>
			<button id="stop" type="button" hidden>Stop</button>
		</div>
	</footer>
	<script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
	}
}

function getNonce(): string {
	let text = '';
	const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
	for (let i = 0; i < 32; i++) {
		text += possible.charAt(Math.floor(Math.random() * possible.length));
	}
	return text;
}
