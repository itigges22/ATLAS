// Webview-side renderer for the ATLAS chat panel. Deliberately dumb: all
// state lives in the extension host (src/ui/chatView.ts); this script only
// renders the messages it is posted, in order. On load it sends {type:
// "ready"} and the host replays the full transcript.

/* global acquireVsCodeApi */

(function () {
	'use strict';

	const vscode = acquireVsCodeApi();

	const messagesEl = document.getElementById('messages');
	const inputEl = document.getElementById('input');
	const sendEl = document.getElementById('send');
	const stopEl = document.getElementById('stop');

	/** The assistant bubble currently receiving streamed text, if any. */
	let openAssistantEl = null;
	/** Tool chips awaiting a tool_result, keyed by tool name (FIFO per name). */
	const pendingChips = new Map();
	/** Open permission cards keyed by prompt id. */
	const permissionCards = new Map();

	function scrollToBottom() {
		messagesEl.scrollTop = messagesEl.scrollHeight;
	}

	function appendBlock(className, text) {
		const el = document.createElement('div');
		el.className = className;
		if (text !== undefined) {
			el.textContent = text;
		}
		messagesEl.appendChild(el);
		scrollToBottom();
		return el;
	}

	function closeAssistantBubble() {
		openAssistantEl = null;
	}

	function assistantBubble() {
		if (!openAssistantEl) {
			openAssistantEl = appendBlock('msg assistant', '');
		}
		return openAssistantEl;
	}

	function addToolChip(name, detail) {
		const chip = document.createElement('div');
		chip.className = 'chip pending';
		const label = document.createElement('span');
		label.className = 'chip-label';
		label.textContent = detail ? name + ' ' + detail : name;
		const status = document.createElement('span');
		status.className = 'chip-status';
		status.textContent = '⋯'; // ⋯ spinner placeholder
		chip.appendChild(status);
		chip.appendChild(label);
		messagesEl.appendChild(chip);
		scrollToBottom();

		if (!pendingChips.has(name)) {
			pendingChips.set(name, []);
		}
		pendingChips.get(name).push(chip);
	}

	function resolveToolChip(tool, success, elapsed, error) {
		const queue = pendingChips.get(tool);
		const chip = queue && queue.length > 0 ? queue.shift() : null;
		if (!chip) {
			// Result without a rendered call (e.g. replay edge) — show standalone.
			appendBlock('chip ' + (success ? 'ok' : 'fail'), (success ? '✓ ' : '✗ ') + tool + (error ? ': ' + error : ''));
			return;
		}
		chip.classList.remove('pending');
		chip.classList.add(success ? 'ok' : 'fail');
		const status = chip.querySelector('.chip-status');
		status.textContent = success ? '✓' : '✗';
		if (elapsed) {
			const time = document.createElement('span');
			time.className = 'chip-elapsed';
			time.textContent = elapsed;
			chip.appendChild(time);
		}
		if (!success && error) {
			const detail = document.createElement('div');
			detail.className = 'chip-error';
			detail.textContent = error;
			chip.appendChild(detail);
		}
	}

	function addPermissionCard(id, tool, detail, message) {
		const card = document.createElement('div');
		card.className = 'permission-card';

		const title = document.createElement('div');
		title.className = 'permission-title';
		title.textContent = 'Permission: ' + tool;
		card.appendChild(title);

		if (message) {
			const body = document.createElement('div');
			body.className = 'permission-message';
			body.textContent = message;
			card.appendChild(body);
		}
		if (detail) {
			const args = document.createElement('div');
			args.className = 'permission-args';
			args.textContent = detail;
			card.appendChild(args);
		}

		const actions = document.createElement('div');
		actions.className = 'permission-actions';
		const buttons = [
			['Allow Once', 'allow-once'],
			['Allow for Session', 'allow-session'],
			['Deny', 'deny'],
		];
		for (const pair of buttons) {
			const button = document.createElement('button');
			button.type = 'button';
			button.textContent = pair[0];
			if (pair[1] === 'deny') {
				button.className = 'deny';
			}
			button.addEventListener('click', () => {
				vscode.postMessage({ type: 'permissionAnswer', id: id, choice: pair[1] });
			});
			actions.appendChild(button);
		}
		card.appendChild(actions);

		messagesEl.appendChild(card);
		permissionCards.set(id, card);
		scrollToBottom();
	}

	function resolvePermissionCard(id, outcome) {
		const card = permissionCards.get(id);
		permissionCards.delete(id);
		if (!card) {
			// Replay path: the prompt message was recorded before its resolution —
			// both replay in order, so a missing card only means a pruned DOM.
			return;
		}
		const actions = card.querySelector('.permission-actions');
		if (actions) {
			actions.remove();
		}
		const result = document.createElement('div');
		result.className = 'permission-outcome';
		result.textContent = outcome;
		card.classList.add('resolved');
		card.appendChild(result);
	}

	function setBusy(busy) {
		sendEl.hidden = busy;
		stopEl.hidden = !busy;
		inputEl.disabled = busy;
	}

	function resetAll() {
		messagesEl.textContent = '';
		pendingChips.clear();
		permissionCards.clear();
		openAssistantEl = null;
	}

	window.addEventListener('message', (event) => {
		const message = event.data;
		switch (message.type) {
			case 'userMessage':
				closeAssistantBubble();
				appendBlock('msg user', message.text);
				break;
			case 'assistantDelta':
				assistantBubble().textContent += message.text;
				scrollToBottom();
				break;
			case 'toolCall':
				closeAssistantBubble();
				addToolChip(message.name, message.detail);
				break;
			case 'toolResult':
				resolveToolChip(message.tool, message.success, message.elapsed, message.error);
				break;
			case 'note':
				closeAssistantBubble();
				appendBlock('note', message.text);
				break;
			case 'permissionPrompt':
				closeAssistantBubble();
				addPermissionCard(message.id, message.tool, message.detail, message.message);
				break;
			case 'permissionResolved':
				resolvePermissionCard(message.id, message.outcome);
				break;
			case 'turnDone':
				closeAssistantBubble();
				break;
			case 'turnError':
				closeAssistantBubble();
				appendBlock('error-card', message.message);
				break;
			case 'busy':
				setBusy(message.value);
				break;
			case 'reset':
				resetAll();
				break;
		}
	});

	function submit() {
		const text = inputEl.value.trim();
		if (text === '') {
			return;
		}
		inputEl.value = '';
		vscode.postMessage({ type: 'submit', text: text });
	}

	sendEl.addEventListener('click', submit);
	stopEl.addEventListener('click', () => vscode.postMessage({ type: 'cancel' }));
	inputEl.addEventListener('keydown', (event) => {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			submit();
		}
	});

	vscode.postMessage({ type: 'ready' });
})();
