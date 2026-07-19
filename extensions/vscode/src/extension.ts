import * as vscode from 'vscode';
import { ChatViewProvider, TOKEN_SECRET_KEY } from './ui/chatView';
import { DiffProvider } from './ui/diffProvider';

export function activate(context: vscode.ExtensionContext) {
	const diffs = new DiffProvider();
	const chat = new ChatViewProvider(context.extensionUri, context.secrets, diffs);

	context.subscriptions.push(
		diffs.register(),

		vscode.window.registerWebviewViewProvider(ChatViewProvider.viewType, chat, {
			webviewOptions: { retainContextWhenHidden: true },
		}),

		vscode.commands.registerCommand('atlas.openChat', () => {
			void vscode.commands.executeCommand(`${ChatViewProvider.viewType}.focus`);
		}),

		vscode.commands.registerCommand('atlas.cancelTurn', () => {
			chat.cancelTurn();
		}),

		vscode.commands.registerCommand('atlas.newConversation', () => {
			chat.newConversation();
		}),

		vscode.commands.registerCommand('atlas.setToken', async () => {
			const token = await vscode.window.showInputBox({
				title: 'ATLAS: Set Service Token',
				prompt: 'Bearer token for the ATLAS proxy (leave empty to clear).',
				password: true,
				ignoreFocusOut: true,
			});
			if (token === undefined) {
				return; // dismissed
			}
			if (token === '') {
				await context.secrets.delete(TOKEN_SECRET_KEY);
				void vscode.window.showInformationMessage('ATLAS: service token cleared.');
			} else {
				await context.secrets.store(TOKEN_SECRET_KEY, token);
				void vscode.window.showInformationMessage('ATLAS: service token saved to Secret Storage.');
			}
		}),
	);
}

export function deactivate() {}
