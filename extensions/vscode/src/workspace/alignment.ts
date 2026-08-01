// Is the proxy's bind the folder the user has open?
//
// When it is not, every tool call is truthful and useless: the agent lists,
// reads and edits a directory the user is not looking at. The passive
// heuristic in mismatch.ts only notices AFTER an edit lands invisibly, which
// costs a turn to learn.
//
// `atlas workspace` already answers this exactly — exit 0 when the open folder
// is covered by the bind, 1 when it is not — so this asks it rather than
// reimplementing the docker inspect in TypeScript.
//
// Deliberately vscode-free (the runner is injected) so it runs under plain
// vitest, same as mismatch.ts.

import { exec } from 'node:child_process';

export type AlignState = 'aligned' | 'misaligned' | 'unknown';

/** Runs a command in cwd and resolves its exit code. Rejects if it cannot spawn. */
export type Runner = (command: string, cwd: string) => Promise<number>;

export function defaultRunner(command: string, cwd: string): Promise<number> {
	return new Promise((resolve, reject) => {
		const child = exec(command, { cwd, timeout: 15_000 }, (err) => {
			if (err && typeof err.code !== 'number') {
				reject(err); // spawn failure: atlas not on PATH
				return;
			}
			resolve(err ? (err.code as number) : 0);
		});
		child.on('error', reject);
	});
}

/** 'unknown' when the CLI is missing or fails in a way we cannot interpret — a
 * user without `atlas` on PATH must never be nagged about alignment. */
export async function checkAlignment(cwd: string, run: Runner = defaultRunner): Promise<AlignState> {
	try {
		const code = await run('atlas workspace', cwd);
		if (code === 0) {
			return 'aligned';
		}
		return code === 1 ? 'misaligned' : 'unknown';
	} catch {
		return 'unknown';
	}
}
