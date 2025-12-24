"use client";
import { useState, useEffect, useRef, useCallback } from "react";

const WS_BASE = "ws://localhost:8000/api";

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

export interface WebSocketMessage {
    type: string;
    data?: unknown;
    timestamp?: number;
}

interface UseWebSocketOptions {
    /** Auto-reconnect on disconnect */
    autoReconnect?: boolean;
    /** Reconnect delay in ms */
    reconnectDelay?: number;
    /** Max reconnect attempts */
    maxReconnectAttempts?: number;
    /** Callback on message received */
    onMessage?: (message: WebSocketMessage) => void;
}

export function useWebSocket(
    endpoint: string = "/ws/events",
    options: UseWebSocketOptions = {}
) {
    const {
        autoReconnect = true,
        reconnectDelay = 2000,
        maxReconnectAttempts = 10,
        onMessage,
    } = options;

    const [status, setStatus] = useState<ConnectionStatus>("disconnected");
    const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectAttemptsRef = useRef(0);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const connect = useCallback(() => {
        // Clear any pending reconnect
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        // Close existing connection
        if (wsRef.current) {
            wsRef.current.close();
        }

        setStatus("connecting");
        const url = `${WS_BASE}${endpoint}`;

        try {
            const ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onopen = () => {
                setStatus("connected");
                reconnectAttemptsRef.current = 0;
            };

            ws.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data);
                    setLastMessage(message);
                    onMessage?.(message);
                } catch {
                    console.warn("Failed to parse WebSocket message:", event.data);
                }
            };

            ws.onerror = () => {
                setStatus("error");
            };

            ws.onclose = () => {
                setStatus("disconnected");
                wsRef.current = null;

                // Auto-reconnect logic
                if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
                    reconnectAttemptsRef.current++;
                    reconnectTimeoutRef.current = setTimeout(() => {
                        connect();
                    }, reconnectDelay);
                }
            };
        } catch (error) {
            console.error("WebSocket connection error:", error);
            setStatus("error");
        }
    }, [endpoint, autoReconnect, reconnectDelay, maxReconnectAttempts, onMessage]);

    const disconnect = useCallback(() => {
        // Prevent auto-reconnect
        reconnectAttemptsRef.current = maxReconnectAttempts;
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setStatus("disconnected");
    }, [maxReconnectAttempts]);

    const sendMessage = useCallback((message: string | object) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            const data = typeof message === "string" ? message : JSON.stringify(message);
            wsRef.current.send(data);
        }
    }, []);

    // Connect on mount
    useEffect(() => {
        connect();
        return () => {
            disconnect();
        };
    }, [connect, disconnect]);

    return {
        status,
        lastMessage,
        connect,
        disconnect,
        sendMessage,
        isConnected: status === "connected",
    };
}
