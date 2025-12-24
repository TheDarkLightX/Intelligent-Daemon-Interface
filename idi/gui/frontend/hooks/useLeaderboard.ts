"use client";
import { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/api";
import { useWebSocket, WebSocketMessage } from "./useWebSocket";

export interface LeaderboardEntry {
    pack_hash: string;
    score: number;
    timestamp_ms: number;
    contributor_id?: string;
    metrics: {
        reward?: number;
        risk?: number;
        complexity?: number;
        episodes_run: number;
    };
}

interface ContributionAcceptedData {
    goal_id: string;
    pack_hash: string;
    contributor_id: string;
    score: number;
    log_index: number;
    leaderboard_position: number | null;
    is_new_leader: boolean;
    metrics: {
        reward: number;
        risk: number;
        complexity: number;
    };
}

interface LeaderboardUpdatedData {
    goal_id: string;
    entries: LeaderboardEntry[];
    active_policy_hash: string | null;
}

interface UseLeaderboardOptions {
    /** Called when a new contribution is accepted */
    onContributionAccepted?: (data: ContributionAcceptedData) => void;
    /** Called when leaderboard updates */
    onLeaderboardUpdated?: (data: LeaderboardUpdatedData) => void;
}

export function useLeaderboard(options: UseLeaderboardOptions = {}) {
    const { onContributionAccepted, onLeaderboardUpdated } = options;

    const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
    const [loading, setLoading] = useState(true);
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Handle WebSocket messages
    const handleMessage = useCallback(
        (message: WebSocketMessage) => {
            if (message.type === "leaderboard_updated") {
                const data = message.data as LeaderboardUpdatedData;
                setEntries(data.entries);
                setLastUpdated(new Date());
                onLeaderboardUpdated?.(data);
            } else if (message.type === "contribution_accepted") {
                const data = message.data as ContributionAcceptedData;
                onContributionAccepted?.(data);
                // Refresh full leaderboard to ensure consistency
                refresh();
            }
        },
        [onContributionAccepted, onLeaderboardUpdated]
    );

    const { status, isConnected } = useWebSocket("/ws/events", {
        onMessage: handleMessage,
    });

    // Initial fetch
    const refresh = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const data = (await api.leaderboard.get()) as LeaderboardEntry[];
            setEntries(data);
            setLastUpdated(new Date());
        } catch (e) {
            setError(e instanceof Error ? e.message : "Failed to load leaderboard");
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        refresh();
    }, [refresh]);

    return {
        entries,
        loading,
        lastUpdated,
        error,
        refresh,
        isLive: isConnected,
        connectionStatus: status,
    };
}
