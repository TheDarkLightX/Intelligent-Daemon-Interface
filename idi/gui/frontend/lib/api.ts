const API_BASE = "http://localhost:8000/api";
const WS_BASE = "ws://localhost:8000/api";

export const api = {
    wizard: {
        getState: async () => {
            const res = await fetch(`${API_BASE}/wizard/state`);
            return res.json();
        },
        next: async (stepData: any) => {
            const res = await fetch(`${API_BASE}/wizard/next`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ step_data: stepData }),
            });
            return res.json();
        },
        prev: async () => {
            const res = await fetch(`${API_BASE}/wizard/prev`, { method: "POST" });
            return res.json();
        },
        reset: async () => {
            const res = await fetch(`${API_BASE}/wizard/reset`, { method: "POST" });
            return res.json();
        },
        getSpec: async () => {
            const res = await fetch(`${API_BASE}/wizard/spec`);
            return res.json();
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
            return res.json();
        },
        stop: async () => {
            const res = await fetch(`${API_BASE}/trainer/stop`, { method: "POST" });
            return res.json();
        },
        getWsUrl: () => `${WS_BASE}/ws/trainer`
    },
    agents: {
        list: async () => {
            const res = await fetch(`${API_BASE}/agents`);
            return res.json();
        },
        save: async (name: string) => {
            const res = await fetch(`${API_BASE}/agents/save`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name }),
            });
            return res.json();
        },
        load: async (name: string) => {
            const res = await fetch(`${API_BASE}/agents/${name}/load`, { method: "POST" });
            return res.json();
        },
        delete: async (name: string) => {
            const res = await fetch(`${API_BASE}/agents/${name}`, { method: "DELETE" });
            return res.json();
        }
    },
    settings: {
        get: async () => {
            const res = await fetch(`${API_BASE}/settings`);
            return res.json();
        },
        update: async (settings: any) => {
            const res = await fetch(`${API_BASE}/settings`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(settings),
            });
            return res.json();
        }
    },
    leaderboard: {
        get: async () => {
            const res = await fetch(`${API_BASE}/leaderboard`);
            return res.json();
        }
    },
    packs: {
        list: async () => {
            const res = await fetch(`${API_BASE}/packs`);
            return res.json();
        },
        install: async (packId: string) => {
            const res = await fetch(`${API_BASE}/packs/${packId}/install`, { method: "POST" });
            return res.json();
        }
    }
};
