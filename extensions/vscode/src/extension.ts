import * as vscode from 'vscode';

// Command IDs contributed in package.json. Real implementations land with the
// chat view (atlas.openChat / atlas.newConversation), turn manager
// (atlas.cancelTurn), and client auth (atlas.setToken) in upcoming commits.
const COMMANDS = ['atlas.openChat', 'atlas.cancelTurn', 'atlas.setToken', 'atlas.newConversation'] as const;

export function activate(context: vscode.ExtensionContext) {
	for (const command of COMMANDS) {
		context.subscriptions.push(
			vscode.commands.registerCommand(command, () => {
				void vscode.window.showInformationMessage(`ATLAS: '${command}' is not implemented yet (scaffold).`);
			}),
		);
	}
}

export function deactivate() {}
