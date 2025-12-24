const API_BASE = "http://localhost:8000/api";
const WS_BASE = "ws://localhost:8000/api";

export class ApiError extends Error {
    status: number;
    detail: string;

    constructor(status: number, detail: string) {
        super(`API Error ${status}: ${detail}`);
        this.name = "ApiError";
        this.status = status;
        this.detail = detail;
    }
}

async function handleResponse<T>(res: Response): Promise<T> {
    if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
            const body = await res.json();
            detail = body.detail || body.message || JSON.stringify(body);
        } catch {
            detail = res.statusText || detail;
        }
        throw new ApiError(res.status, detail);
    }
    return res.json();
}

export const api = {
    wizard: {
        getState: async () => {
            const res = await fetch(`${API_BASE}/wizard/state`);
            return handleResponse(res);
        },
        next: async (stepData: any) => {
            const res = await fetch(`${API_BASE}/wizard/next`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ step_data: stepData }),
            });
            return handleResponse(res);
        },
        prev: async () => {
            const res = await fetch(`${API_BASE}/wizard/prev`, { method: "POST" });
            return handleResponse(res);
        },
        reset: async () => {
            const res = await fetch(`${API_BASE}/wizard/reset`, { method: "POST" });
            return handleResponse(res);
        },
        getSpec: async () => {
            const res = await fetch(`${API_BASE}/wizard/spec`);
            return handleResponse(res);
        },
        export: () => `${API_BASE}/wizard/export`,
    },
    trainer: {
        start: async (config: any, useCrypto: boolean, simConfig: any = {}) => {
            const res = await fetch(`${API_BASE}/trainer/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ config, use_crypto: useCrypto, sim_config: simConfig }),
            });
            return handleResponse(res);
        },
        stop: async () => {
            const res = await fetch(`${API_BASE}/trainer/stop`, { method: "POST" });
            return handleResponse(res);
        },
        getWsUrl: () => `${WS_BASE}/ws/trainer`
    },
    agents: {
        list: async () => {
            const res = await fetch(`${API_BASE}/agents`);
            return handleResponse(res);
        },
        save: async (name: string) => {
            const res = await fetch(`${API_BASE}/agents/save`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name }),
            });
            return handleResponse(res);
        },
        load: async (name: string) => {
            const res = await fetch(`${API_BASE}/agents/${name}/load`, { method: "POST" });
            return handleResponse(res);
        },
        delete: async (name: string) => {
            const res = await fetch(`${API_BASE}/agents/${name}`, { method: "DELETE" });
            return handleResponse(res);
        }
    },
    settings: {
        get: async () => {
            const res = await fetch(`${API_BASE}/settings`);
            return handleResponse(res);
        },
        update: async (settings: any) => {
            const res = await fetch(`${API_BASE}/settings`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(settings),
            });
            return handleResponse(res);
        }
    },
    leaderboard: {
        get: async () => {
            const res = await fetch(`${API_BASE}/leaderboard`);
            return handleResponse(res);
        }
    },
    events: {
        getWsUrl: () => `${WS_BASE}/ws/events`
    },
    packs: {
        list: async () => {
            const res = await fetch(`${API_BASE}/packs`);
            return handleResponse(res);
        },
        install: async (packId: string) => {
            const res = await fetch(`${API_BASE}/packs/${packId}/install`, { method: "POST" });
            return handleResponse(res);
        }
    }
};
