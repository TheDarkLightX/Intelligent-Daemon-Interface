"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Play, Square, Activity, Timer, Check, RotateCcw, Info } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { cn } from "@/lib/utils";
import { useToast } from "@/components/ui/toast";


type TrainingStats = {
    episode: number;
    total_episodes: number;
    last_reward: number;
    status: "starting" | "running" | "completed" | "cancelled" | "idle";
    final_stats?: any;
    history?: { time: number, value: number }[];
};

export function TrainingMonitor() {
    const [status, setStatus] = useState<"idle" | "running" | "completed">("idle");
    const [episodeData, setEpisodeData] = useState<any[]>([]); // Reward history
    const [priceData, setPriceData] = useState<any[]>([]); // Price history of last episode
    const [currentStats, setCurrentStats] = useState<TrainingStats | null>(null);
    const ws = useRef<WebSocket | null>(null);
    const [useCrypto, setUseCrypto] = useState(false);
    const [startTime, setStartTime] = useState<number | null>(null);
    const [elapsedSeconds, setElapsedSeconds] = useState(0);
    const toast = useToast();

    // Sim Params
    const [volatility, setVolatility] = useState(0.01);
    const [drift, setDrift] = useState(0.002);
    const [fee, setFee] = useState(5.0);

    // Default presets
    const DEFAULT_VOLATILITY = 0.01;
    const DEFAULT_DRIFT = 0.002;
    const DEFAULT_FEE = 5.0;

    useEffect(() => {
        loadSettings();
        connectWs();
        // Elapsed time timer
        const timer = setInterval(() => {
            if (startTime) {
                setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
            }
        }, 1000);

        return () => {
            if (ws.current) ws.current.close();
            clearInterval(timer);
        };
    }, []);

    const loadSettings = async () => {
        try {
            const settings = (await api.settings.get()) as { market_sim?: { volatility?: number; drift_bull?: number; fee_bps?: number } };
            if (settings?.market_sim) {
                setVolatility(settings.market_sim.volatility ?? DEFAULT_VOLATILITY);
                setDrift(settings.market_sim.drift_bull ?? DEFAULT_DRIFT);
                setFee(settings.market_sim.fee_bps ?? DEFAULT_FEE);
            }
        } catch (e) {
            console.error("Failed to load settings");
        }
    };

    const connectWs = () => {
        const url = api.trainer.getWsUrl();
        const socket = new WebSocket(url);

        socket.onopen = () => {
            console.log("Connected to Trainer WS");
        };

        socket.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            setCurrentStats(msg);

            if (msg.status === "running") {
                setStatus("running");
                // Update Reward Chart
                setEpisodeData(prev => {
                    const newData = [...prev, { episode: msg.episode, reward: msg.last_reward }];
                    if (newData.length > 100) return newData.slice(newData.length - 100);
                    return newData;
                });

                // Update Price Chart if history available
                if (msg.history) {
                    setPriceData(msg.history);
                }
            } else if (msg.status === "completed") {
                setStatus("completed");
                toast.success("Training completed successfully!");
            } else if (msg.status === "cancelled") {
                setStatus("idle");
                setStartTime(null);
                toast.info("Training cancelled");
            }
        };

        socket.onclose = () => {
            setTimeout(connectWs, 2000); // Reconnect
        };

        ws.current = socket;
    };

    const handleStart = async () => {
        try {
            setEpisodeData([]);
            setPriceData([]);
            setStartTime(Date.now());
            setElapsedSeconds(0);
            const simConfig = useCrypto ? {
                vol_base: parseFloat(volatility.toString()),
                drift_bull: parseFloat(drift.toString()),
                drift_bear: -parseFloat(drift.toString()),
                fee_bps: parseFloat(fee.toString())
            } : {};

            await api.trainer.start({}, useCrypto, simConfig);
            setStatus("running");
            toast.info("Training started");
        } catch (e) {
            console.error("Failed to start training", e);
            toast.error("Failed to start training");
        }
    };

    const handleStop = async () => {
        try {
            await api.trainer.stop();
            setStatus("idle");
            setStartTime(null);
        } catch (e) {
            console.error(e);
            toast.error("Failed to stop training");
        }
    };

    const handleResetParams = () => {
        setVolatility(DEFAULT_VOLATILITY);
        setDrift(DEFAULT_DRIFT);
        setFee(DEFAULT_FEE);
        toast.info("Parameters reset to defaults");
    };

    const formatElapsed = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Controls */}
            <Card className="glass md:col-span-1 h-fit">
                <CardHeader>
                    <CardTitle>Controls</CardTitle>
                    <CardDescription>Manage training session</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex items-center space-x-2 p-3 border rounded bg-secondary/20">
                        <input
                            type="checkbox"
                            checked={useCrypto}
                            onChange={(e) => setUseCrypto(e.target.checked)}
                            id="crypto-toggle"
                            className="w-4 h-4"
                        />
                        <label htmlFor="crypto-toggle" className="text-sm cursor-pointer select-none">Use Crypto Market Sim</label>
                    </div>

                    {useCrypto && (
                        <div className="space-y-3 p-3 border rounded bg-secondary/10">
                            <h4 className="text-xs font-semibold uppercase text-muted-foreground">Sim Parameters</h4>

                            <div className="space-y-1">
                                <label className="text-xs">Base Volatility</label>
                                <div className="flex items-center gap-2">
                                    <input
                                        type="range" min="0.001" max="0.1" step="0.001"
                                        value={volatility} onChange={(e) => setVolatility(parseFloat(e.target.value))}
                                        className="flex-1"
                                    />
                                    <span className="text-xs font-mono w-12 text-right">{volatility.toFixed(3)}</span>
                                </div>
                            </div>

                            <div className="space-y-1">
                                <label className="text-xs">Drift (Bull)</label>
                                <div className="flex items-center gap-2">
                                    <input
                                        type="range" min="0" max="0.01" step="0.0001"
                                        value={drift} onChange={(e) => setDrift(parseFloat(e.target.value))}
                                        className="flex-1"
                                    />
                                    <span className="text-xs font-mono w-12 text-right">{drift.toFixed(4)}</span>
                                </div>
                            </div>

                            <div className="space-y-1">
                                <label className="text-xs">Fee (bps)</label>
                                <div className="flex items-center gap-2">
                                    <input
                                        type="range" min="0" max="20" step="0.5"
                                        value={fee} onChange={(e) => setFee(parseFloat(e.target.value))}
                                        className="flex-1"
                                    />
                                    <span className="text-xs font-mono w-12 text-right">{fee.toFixed(1)}</span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Reset to Defaults Button */}
                    {useCrypto && (
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={handleResetParams}
                            disabled={status === "running"}
                            className="w-full text-muted-foreground hover:text-foreground"
                        >
                            <RotateCcw className="w-3 h-3 mr-2" /> Reset to Defaults
                        </Button>
                    )}

                    {/* Single Contextual Action Button */}
                    {status === "running" ? (
                        <Button
                            onClick={handleStop}
                            variant="destructive"
                            className="w-full"
                        >
                            <Square className="w-4 h-4 mr-2" /> Stop Training
                        </Button>
                    ) : (
                        <Button
                            onClick={handleStart}
                            className="w-full bg-green-600 hover:bg-green-700"
                        >
                            <Play className="w-4 h-4 mr-2" /> Start Training
                        </Button>
                    )}

                    {/* Status Pill */}
                    <div className="flex items-center justify-between p-2 rounded bg-secondary/20 border border-border/50">
                        <div className="flex items-center gap-2">
                            <div className={cn(
                                "w-2 h-2 rounded-full",
                                status === "running" ? "bg-green-500 animate-pulse" :
                                    status === "completed" ? "bg-blue-500" : "bg-yellow-500"
                            )} />
                            <span className="text-xs font-medium uppercase tracking-wider">
                                {status}
                            </span>
                        </div>
                        {status === "running" && startTime && (
                            <span className="text-xs text-muted-foreground font-mono">
                                <Timer className="w-3 h-3 inline mr-1" />
                                {formatElapsed(elapsedSeconds)}
                            </span>
                        )}
                    </div>

                    {currentStats && (
                        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border">
                            <div>
                                <span className="text-xs text-muted-foreground block">Progress</span>
                                <div className="text-xl font-mono">
                                    {currentStats.episode} <span className="text-xs text-muted-foreground">/ {currentStats.total_episodes}</span>
                                </div>
                            </div>
                            <div>
                                <span className="text-xs text-muted-foreground block">Last Reward</span>
                                <div className={cn(
                                    "text-xl font-mono",
                                    currentStats.last_reward > 0 ? "text-green-400" : "text-red-400"
                                )}>
                                    {currentStats.last_reward.toFixed(2)}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Status Legend */}
                    <div className="pt-3 border-t border-border/50">
                        <div className="flex items-center gap-1 mb-2 text-xs text-muted-foreground">
                            <Info className="w-3 h-3" /> Status Legend
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                            <div className="flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full bg-yellow-500" />
                                <span>Idle</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full bg-green-500" />
                                <span>Running</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full bg-blue-500" />
                                <span>Done</span>
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Price Chart */}
            <Card className="glass md:col-span-2 min-h-[400px]">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Activity className="w-5 h-5 text-violet-400" /> Live Market Replay
                    </CardTitle>
                    <CardDescription>Price action from the latest training episode</CardDescription>
                </CardHeader>
                <CardContent className="h-[350px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={priceData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis
                                dataKey="time"
                                stroke="#666"
                                fontSize={12}
                                tickFormatter={(val) => val + "t"}
                            />
                            <YAxis
                                domain={['auto', 'auto']}
                                stroke="#666"
                                fontSize={12}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: "#000", border: "1px solid #333" }}
                                itemStyle={{ color: "#fff" }}
                            />
                            <Line
                                type="monotone"
                                dataKey="value"
                                stroke="#22c55e"
                                strokeWidth={2}
                                dot={false}
                                animationDuration={0} // fast updates
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </CardContent>
            </Card>

            {/* Reward Chart */}
            <Card className="glass md:col-span-3 min-h-[200px]">
                <CardHeader className="py-3">
                    <CardTitle className="text-sm">Reward History</CardTitle>
                </CardHeader>
                <CardContent className="h-[150px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={episodeData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="episode" hide />
                            <YAxis stroke="#666" fontSize={10} />
                            <Line type="monotone" dataKey="reward" stroke="#8b5cf6" dot={false} strokeWidth={1} />
                        </LineChart>
                    </ResponsiveContainer>
                </CardContent>
            </Card>

            {/* Status Bar */}
            <div className="md:col-span-3">
                <Card className="bg-secondary/20 border-border/50">
                    <CardContent className="p-4 flex items-center gap-4">
                        <div className={cn(
                            "w-3 h-3 rounded-full animate-pulse",
                            status === "running" ? "bg-green-500" : status === "idle" ? "bg-yellow-500" : "bg-blue-500"
                        )} />
                        <span className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
                            Status: {status}
                        </span>
                        {status === "completed" && (
                            <span className="ml-auto text-sm text-green-400 flex items-center gap-2">
                                <Check className="w-4 h-4" /> Training completed successfully
                            </span>
                        )}
                    </CardContent>
                </Card>
            </div>

        </div>
    );
}
