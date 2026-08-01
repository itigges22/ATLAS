import * as vscode from 'vscode';
import { ChatViewProvider, TOKEN_SECRET_KEY } from './ui/chatView';
import { DiffProvider } from './ui/diffProvider';
import { StatusBar } from './ui/statusBar';

export function activate(context: vscode.ExtensionContext) {
	const diffs = new DiffProvider();
	const chat = new ChatViewProvider(context.extensionUri, context.secrets, diffs, context.workspaceState);
	const statusBar = new StatusBar(() => chat.makeClient());
	chat.attachStatusBar(statusBar);

	context.subscriptions.push(
		diffs.register(),
		statusBar,

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

		vscode.commands.registerCommand('atlas.statusMenu', () => {
			void statusBar.showMenu();
		}),

		vscode.commands.registerCommand('atlas.refreshStatus', () => {
			void statusBar.refresh();
		}),

		// Moves the proxy + sandbox binds onto the open folder. It shells out
		// to `atlas workspace align` rather than driving docker from here:
		// that command wraps runtime._align_workspace, the same path `atlas
		// tui` takes on launch, which recreates BOTH containers together.
		// Recreating one alone splits the binds and the agent goes
		// split-brained with every health check still green.
		vscode.commands.registerCommand('atlas.useThisFolder', async () => {
			const folder = vscode.workspace.workspaceFolders?.[0];
			if (!folder) {
				void vscode.window.showWarningMessage('ATLAS: open a folder first.');
				return;
			}
			const terminal = vscode.window.createTerminal({
				name: 'ATLAS: align workspace',
				cwd: folder.uri.fsPath,
			});
			// Visible on purpose: it restarts containers and can take a few
			// seconds, so the user should see it happen rather than wonder
			// whether the editor froze.
			terminal.show();
			terminal.sendText('atlas workspace align');
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
			void statusBar.refresh(); // token change flips 401 state immediately
		}),

		// Poll interval / enabled flag changes take effect without a reload.
		vscode.workspace.onDidChangeConfiguration((event) => {
			if (event.affectsConfiguration('atlas.statusBar')) {
				statusBar.applyConfig();
			}
		}),
	);

	statusBar.start();
}

export function deactivate() {}
