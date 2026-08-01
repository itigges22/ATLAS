// Status bar item fed by a GET /ready poll. Polling pauses while a turn is
// streaming (the SSE connection is proof of connectivity, and /ready hits
// four downstream health checks). /v1/calibration/status is only fetched at
// activation and on explicit user refresh — each call re-probes the lens
// service (docs/API.md).

import * as vscode from 'vscode';
import { AtlasApiError, AtlasClient } from '../client/atlasClient';
import type { ReadyResponse } from '../client/types';

/** Provides the current client config; chatView owns token resolution. */
export type ClientFactory = () => Promise<AtlasClient>;

type State = 'ready' | 'starting' | 'unreachable' | 'unauthorized';

/** Human labels for the /ready gates, in display order. */
const GATE_LABELS: [keyof ReadyResponse, string][] = [
	['inference', 'inference'],
	['lens_ready', 'lens'],
	['sandbox', 'sandbox'],
	['v3', 'v3 service'],
];

export class StatusBar implements vscode.Disposable {
	private readonly item: vscode.StatusBarItem;
	private timer: NodeJS.Timeout | undefined;
	private streaming = false;
	private disposed = false;
	/** Last calibration verdict line, shown in the tooltip. */
	private calibrationLine = '';

	constructor(private readonly clientFactory: ClientFactory) {
		this.item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
		this.item.name = 'ATLAS';
		this.item.command = 'atlas.statusMenu';
		this.render('starting', 'checking…');
	}

	/** Start polling (config-gated) and fetch calibration once. */
	start(): void {
		this.applyConfig();
		void this.refreshCalibration();
	}

	/** Chat view signals: pause the poll while a turn streams. */
	setStreaming(streaming: boolean): void {
		this.streaming = streaming;
		if (!streaming) {
			void this.poll(); // refresh promptly after the turn ends
		}
	}

	/** Re-read config (enabled flag + interval) and restart the timer. */
	applyConfig(): void {
		if (this.timer !== undefined) {
			clearInterval(this.timer);
			this.timer = undefined;
		}
		const config = vscode.workspace.getConfiguration('atlas');
		if (!config.get<boolean>('statusBar.enabled', true)) {
			this.item.hide();
			return;
		}
		this.item.show();
		const intervalSec = Math.max(5, config.get<number>('statusBar.pollIntervalSec', 15));
		this.timer = setInterval(() => void this.poll(), intervalSec * 1000);
		void this.poll();
	}

	/** Manual refresh: /ready now plus a calibration re-probe. */
	async refresh(): Promise<void> {
		await Promise.all([this.poll(), this.refreshCalibration()]);
	}

	private async poll(): Promise<void> {
		if (this.streaming || this.disposed) {
			return;
		}
		let client: AtlasClient;
		try {
			client = await this.clientFactory();
		} catch {
			return;
		}
		try {
			const ready = await client.getReady();
			if (ready.ready) {
				this.render('ready', 'all gates passing');
			} else {
				const failing = GATE_LABELS.filter(([key]) => ready[key] !== true).map(([, label]) => label);
				this.render('starting', failing.length > 0 ? `waiting on: ${failing.join(', ')}` : 'starting…');
			}
		} catch (error) {
			if (error instanceof AtlasApiError && error.status === 401) {
				this.render('unauthorized', 'the proxy rejected the token (401)');
			} else {
				this.render('unreachable', 'the proxy is not reachable');
			}
		}
	}

	private async refreshCalibration(): Promise<void> {
		try {
			const client = await this.clientFactory();
			const status = await client.getCalibrationStatus();
			this.calibrationLine = `lens: ${status.lens.verdict} · asa: ${status.asa.verdict}`;
		} catch {
			this.calibrationLine = ''; // proxy down or pre-auth — tooltip just omits it
		}
		this.rerenderTooltip();
	}

	private lastDetail = '';

	private render(state: State, detail: string): void {
		this.lastDetail = detail;
		switch (state) {
			case 'ready':
				this.item.text = '$(check) ATLAS';
				this.item.backgroundColor = undefined;
				break;
			case 'starting':
				this.item.text = '$(sync~spin) ATLAS';
				this.item.backgroundColor = undefined;
				break;
			case 'unreachable':
				this.item.text = '$(warning) ATLAS';
				this.item.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
				break;
			case 'unauthorized':
				this.item.text = '$(circle-slash) ATLAS';
				this.item.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
				break;
		}
		this.rerenderTooltip();
	}

	private rerenderTooltip(): void {
		const lines = [`ATLAS proxy: ${this.lastDetail}`];
		if (this.calibrationLine !== '') {
			lines.push(this.calibrationLine);
		}
		this.item.tooltip = lines.join('\n');
	}

	/** Status bar click → quick pick of the useful actions. */
	async showMenu(): Promise<void> {
		const picked = await vscode.window.showQuickPick(
			[
				{ label: '$(comment-discussion) Open Chat', action: 'chat' },
				{ label: '$(refresh) Refresh Status & Calibration', action: 'refresh' },
				{ label: '$(gear) ATLAS Settings', action: 'settings' },
			],
			{ placeHolder: this.item.tooltip?.toString() },
		);
		switch (picked?.action) {
			case 'chat':
				void vscode.commands.executeCommand('atlas.openChat');
				break;
			case 'refresh':
				void this.refresh();
				break;
			case 'settings':
				void vscode.commands.executeCommand('workbench.action.openSettings', '@ext:atlas.atlas-vscode');
				break;
		}
	}

	dispose(): void {
		this.disposed = true;
		if (this.timer !== undefined) {
			clearInterval(this.timer);
		}
		this.item.dispose();
	}
}
