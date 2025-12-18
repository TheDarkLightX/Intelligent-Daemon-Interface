"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { CodeBlock } from "@/components/ui/code-block";

import { ArrowLeft, ArrowRight, Save, Bot, ChevronLeft, ChevronRight, Activity, Check, Download, RefreshCw, AlertCircle } from "lucide-react";

import { AgentVisualizer } from "@/components/Wizard/AgentVisualizer";
import { cn } from "@/lib/utils";

// Types matching backend
type WizardData = {
    name: string;
    strategy: string;
    selected_inputs: Record<string, boolean>;
    num_layers: number;
    include_safety: boolean;
    include_communication: boolean;
    ensemble_pattern?: string;
    ensemble_threshold?: number;
};

type WizardState = {
    current_step_idx: number;
    data: WizardData;
    validation_errors: Record<string, string>;
};

const STEPS = ["Strategy", "Inputs", "Layers", "Safety", "Review"];

export function Wizard() {
    const [state, setState] = useState<WizardState | null>(null);
    const [loading, setLoading] = useState(true);
    const [generatedSpec, setGeneratedSpec] = useState<string | null>(null);

    useEffect(() => {
        fetchState();
    }, []);

    const fetchState = async () => {
        try {
            const data = await api.wizard.getState();
            setState(data);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const updateField = (field: keyof WizardData, value: any) => {
        if (!state) return;
        setState({
            ...state,
            data: { ...state.data, [field]: value }
        });
    };

    const handleNext = async () => {
        if (!state) return;
        try {
            setLoading(true);
            const newState = await api.wizard.next(state.data);
            setState(newState);
        } finally {
            setLoading(false);
        }
    };

    const handlePrev = async () => {
        try {
            setLoading(true);
            const newState = await api.wizard.prev();
            setState(newState);
        } finally {
            setLoading(false);
        }
    };

    const handleGenerate = async () => {
        try {
            setLoading(true);
            const res = await api.wizard.getSpec();
            setGeneratedSpec(res.spec);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const handleReset = async () => {
        setGeneratedSpec(null);
        const newState = await api.wizard.reset();
        setState(newState);
    };


    const handleSave = async () => {
        if (!state) return;
        try {
            setLoading(true);
            await api.agents.save(state.data.name);
            alert("Agent saved successfully!");
            handleReset(); // Optionally reset/go back
        } catch (e) {
            console.error(e);
            alert("Failed to save agent");
        } finally {
            setLoading(false);
        }
    };

    if (!state) return <div className="text-center p-8">Loading Wizard...</div>;

    const currentStep = STEPS[state.current_step_idx] || "Unknown";

    return (
        <div className="space-y-6">
            {/* Progress Bar */}
            <div className="flex justify-between mb-8 relative">
                <div className="absolute top-1/2 left-0 w-full h-1 bg-secondary -z-10 rounded-full" />
                <div
                    className="absolute top-1/2 left-0 h-1 bg-cyan-500 -z-10 rounded-full transition-all duration-500"
                    style={{ width: `${(state.current_step_idx / (STEPS.length - 1)) * 100}%` }}
                />
                {STEPS.map((step, idx) => (
                    <div key={step} className="flex flex-col items-center gap-2">
                        <div className={cn(
                            "w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300 border-2",
                            idx <= state.current_step_idx
                                ? "bg-background border-cyan-500 text-cyan-500"
                                : "bg-secondary border-transparent text-muted-foreground"
                        )}>
                            {idx < state.current_step_idx ? <Check className="w-4 h-4" /> : idx + 1}
                        </div>
                        <span className={cn(
                            "text-xs font-medium transition-colors",
                            idx <= state.current_step_idx ? "text-cyan-400" : "text-muted-foreground"
                        )}>{step}</span>
                    </div>
                ))}
            </div>

            <AnimatePresence mode="wait">
                {generatedSpec ? (
                    <motion.div
                        key="result"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                    >
                        <Card className="glass border-green-500/30">
                            <CardHeader>
                                <CardTitle className="text-green-400">Agent Spec Generated!</CardTitle>
                                <CardDescription>Your Tau Agent specification is ready.</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <pre className="bg-black/50 p-4 rounded-md overflow-x-auto text-xs font-mono text-green-300 max-h-[500px]">
                                    {generatedSpec}
                                </pre>
                            </CardContent>
                            <CardFooter className="flex gap-4">
                                <Button onClick={handleSave} className="flex-1 bg-green-600 hover:bg-green-700">
                                    <Bot className="w-4 h-4 mr-2" /> Save to Library
                                </Button>
                                <Button onClick={handleReset} variant="outline" className="flex-1 gap-2">
                                    <RefreshCw className="w-4 h-4" /> Create Another
                                </Button>
                            </CardFooter>
                        </Card>
                    </motion.div>
                ) : (
                    <motion.div
                        key={state.current_step_idx}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.2 }}
                    >
                        <Card className="glass min-h-[400px] flex flex-col">
                            <CardHeader>
                                <CardTitle>{currentStep}</CardTitle>
                                <CardDescription>Configure your agent parameters.</CardDescription>
                            </CardHeader>
                            <CardContent className="flex-1 space-y-6">
                                {/* Validation Errors */}
                                {Object.keys(state.validation_errors).length > 0 && (
                                    <div className="bg-destructive/10 border border-destructive/20 text-destructive p-3 rounded-md text-sm flex items-center gap-2">
                                        <AlertCircle className="w-4 h-4" />
                                        Please fix the errors below.
                                    </div>
                                )}

                                {/* Step Content */}
                                {state.current_step_idx === 0 && (
                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <label className="text-sm font-medium">Agent Name</label>
                                            <input
                                                type="text"
                                                value={state.data.name}
                                                onChange={(e) => updateField("name", e.target.value)}
                                                className="w-full bg-secondary/50 border border-input rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                                                placeholder="e.g. MySuperAgent"
                                            />
                                            {state.validation_errors.name && <p className="text-xs text-destructive">{state.validation_errors.name}</p>}
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-sm font-medium">Strategy</label>
                                            <div className="grid grid-cols-2 gap-4">
                                                {["momentum", "mean_reversion", "regime_aware", "custom", "ensemble"].map((strat) => (
                                                    <div
                                                        key={strat}
                                                        onClick={() => updateField("strategy", strat)}
                                                        className={cn(
                                                            "p-4 rounded-lg border cursor-pointer transition-all hover:bg-secondary",
                                                            state.data.strategy === strat ? "border-cyan-500 bg-cyan-500/10" : "border-border bg-card"
                                                        )}
                                                    >
                                                        <div className="font-semibold capitalize">{strat.replace("_", " ")}</div>
                                                        <div className="text-xs text-muted-foreground mt-1">
                                                            {strat === "momentum" ? "Follows market trends" :
                                                                strat === "mean_reversion" ? "Bets against trends" :
                                                                    strat === "ensemble" ? "Combines multiple agents" : "Custom strategy"}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {state.current_step_idx === 1 && (
                                    <div className="space-y-4">
                                        <p className="text-sm text-muted-foreground">Select input streams relative to the agent.</p>
                                        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                                            {["q_buy", "q_sell", "price_up", "price_down", "trend", "volume", "regime", "risk_budget_ok"].map((input) => (
                                                <div
                                                    key={input}
                                                    onClick={() => {
                                                        const newInputs = { ...state.data.selected_inputs, [input]: !state.data.selected_inputs[input] };
                                                        updateField("selected_inputs", newInputs);
                                                    }}
                                                    className={cn(
                                                        "p-3 rounded-md border cursor-pointer flex items-center gap-3 transition-all",
                                                        state.data.selected_inputs[input] ? "border-cyan-500 bg-cyan-500/10" : "border-border"
                                                    )}
                                                >
                                                    <div className={cn(
                                                        "w-4 h-4 rounded border flex items-center justify-center",
                                                        state.data.selected_inputs[input] ? "bg-cyan-500 border-cyan-500" : "border-muted-foreground"
                                                    )}>
                                                        {state.data.selected_inputs[input] && <Check className="w-3 h-3 text-black" />}
                                                    </div>
                                                    <span className="text-sm font-mono">{input}</span>
                                                </div>
                                            ))}
                                        </div>
                                        {state.validation_errors.inputs && <p className="text-xs text-destructive">{state.validation_errors.inputs}</p>}
                                    </div>
                                )}

                                {state.current_step_idx === 2 && (
                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <label className="text-sm font-medium">Number of Layers</label>
                                            <input
                                                type="number"
                                                value={state.data.num_layers}
                                                onChange={(e) => updateField("num_layers", parseInt(e.target.value))}
                                                className="w-full bg-secondary/50 border border-input rounded-md px-3 py-2 text-sm"
                                                min={1} max={10}
                                            />
                                        </div>
                                        <div className="p-4 rounded-lg bg-secondary/30 border border-border">
                                            <h4 className="font-semibold text-sm mb-2">Layer Configuration</h4>
                                            <p className="text-xs text-muted-foreground">
                                                Layers allow the agent to process information hierarchically.
                                                More layers increase complexity but allow for more sophisticated behaviors.
                                            </p>
                                        </div>
                                    </div>
                                )}

                                {state.current_step_idx === 3 && (
                                    <div className="space-y-4">
                                        <div
                                            onClick={() => updateField("include_safety", !state.data.include_safety)}
                                            className={cn(
                                                "p-4 rounded-lg border cursor-pointer flex items-center justify-between",
                                                state.data.include_safety ? "border-green-500 bg-green-500/10" : "border-border"
                                            )}
                                        >
                                            <div>
                                                <div className="font-semibold">Include Safety Rails</div>
                                                <div className="text-xs text-muted-foreground">Prevents catastrophic actions</div>
                                            </div>
                                            {state.data.include_safety && <Check className="text-green-500" />}
                                        </div>

                                        <div
                                            onClick={() => updateField("include_communication", !state.data.include_communication)}
                                            className={cn(
                                                "p-4 rounded-lg border cursor-pointer flex items-center justify-between",
                                                state.data.include_communication ? "border-purple-500 bg-purple-500/10" : "border-border"
                                            )}
                                        >
                                            <div>
                                                <div className="font-semibold">Enable Communication</div>
                                                <div className="text-xs text-muted-foreground">Allow agent to emote and signal</div>
                                            </div>
                                            {state.data.include_communication && <Check className="text-purple-500" />}
                                        </div>
                                    </div>
                                )}

                                {state.current_step_idx === 4 && (
                                    <div className="space-y-4">
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="p-3 bg-secondary/20 rounded border border-border">
                                                <span className="text-xs text-muted-foreground block">Name</span>
                                                <span className="font-mono">{state.data.name}</span>
                                            </div>
                                            <div className="p-3 bg-secondary/20 rounded border border-border">
                                                <span className="text-xs text-muted-foreground block">Strategy</span>
                                                <span className="capitalize">{state.data.strategy}</span>
                                            </div>
                                            <div className="p-3 bg-secondary/20 rounded border border-border col-span-2">
                                                <span className="text-xs text-muted-foreground block">Inputs</span>
                                                <div className="flex flex-wrap gap-2 mt-1">
                                                    {Object.entries(state.data.selected_inputs).filter(([_, v]) => v).map(([k]) => (
                                                        <span key={k} className="text-xs bg-cyan-500/20 text-cyan-300 px-2 py-1 rounded-full font-mono">{k}</span>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>

                                        {/* Visualizer */}
                                        <div className="mt-6">
                                            <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                                                <Activity className="w-4 h-4 text-cyan-400" /> Logic Visualization
                                            </h4>
                                            <AgentVisualizer
                                                inputs={state.data.selected_inputs}
                                                strategy={state.data.strategy}
                                                layers={state.data.num_layers}
                                            />
                                        </div>
                                    </div>
                                )}

                            </CardContent>
                            <CardFooter className="flex justify-between border-t border-border/50 p-6">
                                <Button
                                    variant="outline"
                                    onClick={handlePrev}
                                    disabled={state.current_step_idx === 0 || loading}
                                >
                                    <ChevronLeft className="w-4 h-4 mr-2" /> Back
                                </Button>

                                {state.current_step_idx === STEPS.length - 1 ? (
                                    <Button
                                        onClick={handleGenerate}
                                        className="bg-cyan-500 hover:bg-cyan-600 text-white"
                                        disabled={loading}
                                    >
                                        {loading ? "Generating..." : "Generate Agent Spec"}
                                    </Button>
                                ) : (
                                    <Button
                                        onClick={handleNext}
                                        className="bg-primary hover:bg-primary/90"
                                        disabled={loading}
                                    >
                                        Next <ChevronRight className="w-4 h-4 ml-2" />
                                    </Button>
                                )}

                                {state.current_step_idx > 3 && (
                                    <Button
                                        variant="outline"
                                        onClick={() => window.open(api.wizard.export(), "_blank")}
                                        className="ml-2 border-dashed"
                                        title="Download .tau file"
                                    >
                                        <Download className="w-4 h-4" />
                                    </Button>
                                )}
                            </CardFooter>
                        </Card>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
