"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Save, Settings as SettingsIcon, RotateCcw } from "lucide-react";
import { useToast } from "@/components/ui/toast";

const DEFAULT_SETTINGS = {
    market_sim: {
        volatility: 0.01,
        drift_bull: 0.002,
        drift_bear: -0.002,
        fee_bps: 5.0
    }
};

export default function SettingsPage() {
    const [settings, setSettings] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const toast = useToast();

    useEffect(() => {
        loadSettings();
    }, []);

    const loadSettings = async () => {
        try {
            const data = await api.settings.get();
            setSettings(data);
        } catch (e) {
            console.error(e);
            toast.error("Failed to load settings");
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        try {
            await api.settings.update(settings);
            toast.success("Settings saved!");
        } catch (e) {
            toast.error("Failed to save settings");
        }
    };

    const handleReset = () => {
        setSettings({ ...settings, market_sim: DEFAULT_SETTINGS.market_sim });
        toast.info("Settings reset to defaults (not yet saved)");
    };

    const updateMarket = (key: string, value: number) => {
        setSettings({
            ...settings,
            market_sim: {
                ...settings.market_sim,
                [key]: value
            }
        });
    };

    if (loading) return <div className="p-8 text-center text-muted-foreground">Loading...</div>;

    const Input = (props: any) => (
        <input
            {...props}
            className="flex h-10 w-full rounded-md border bg-secondary/20 border-border px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 text-slate-100"
        />
    );

    const Label = ({ children }: { children: React.ReactNode }) => (
        <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 text-slate-300 mb-2 block">
            {children}
        </label>
    );

    return (
        <div className="max-w-2xl mx-auto space-y-6">
            <div>
                <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-100 to-slate-400">
                    Settings
                </h1>
                <p className="text-muted-foreground mt-2">Configure global preferences.</p>
            </div>

            <Card className="glass">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <SettingsIcon className="w-5 h-5 text-cyan-400" /> Market Simulation Defaults
                    </CardTitle>
                    <CardDescription>Default parameters for the Crypto Market environment.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="space-y-2">
                        <Label>Base Volatility</Label>
                        <div className="flex gap-4 items-center">
                            <Input
                                type="number" step="0.001"
                                value={settings?.market_sim?.volatility || 0.01}
                                onChange={(e: any) => updateMarket("volatility", parseFloat(e.target.value))}
                            />
                        </div>
                    </div>
                    <div className="space-y-2">
                        <Label>Drift (Bull)</Label>
                        <Input
                            type="number" step="0.0001"
                            value={settings?.market_sim?.drift_bull || 0.002}
                            onChange={(e: any) => updateMarket("drift_bull", parseFloat(e.target.value))}
                        />
                    </div>
                    <div className="space-y-2">
                        <Label>Drift (Bear)</Label>
                        <Input
                            type="number" step="0.0001"
                            value={settings?.market_sim?.drift_bear || -0.002}
                            onChange={(e: any) => updateMarket("drift_bear", parseFloat(e.target.value))}
                        />
                    </div>
                    <div className="space-y-2">
                        <Label>Fee (bps)</Label>
                        <Input
                            type="number" step="0.1"
                            value={settings?.market_sim?.fee_bps || 5.0}
                            onChange={(e: any) => updateMarket("fee_bps", parseFloat(e.target.value))}
                        />
                    </div>
                </CardContent>
            </Card>

            <div className="flex justify-between">
                <Button variant="outline" onClick={handleReset}>
                    <RotateCcw className="w-4 h-4 mr-2" /> Reset to Defaults
                </Button>
                <Button onClick={handleSave} className="bg-cyan-500 hover:bg-cyan-600 text-white">
                    <Save className="w-4 h-4 mr-2" /> Save Changes
                </Button>
            </div>
        </div>
    );
}

