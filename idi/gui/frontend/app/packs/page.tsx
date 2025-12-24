"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Package, Download, Check } from "lucide-react";
import { useToast } from "@/components/ui/toast";
import { useConfirmDialog } from "@/components/ui/confirm-dialog";

type Pack = {
    id: string;
    name: string;
    description: string;
    price: number;
    installed?: boolean;
    config: {
        episodes: number;
        learning_rate: number;
    };
};

export default function PacksPage() {
    const [packs, setPacks] = useState<Pack[]>([]);
    const [loading, setLoading] = useState(true);
    const [installedIds, setInstalledIds] = useState<Set<string>>(new Set());
    const toast = useToast();
    const { confirm } = useConfirmDialog();

    useEffect(() => {
        loadPacks();
    }, []);

    const loadPacks = async () => {
        try {
            const data = await api.packs.list() as Pack[];
            setPacks(data);
        } catch (e) {
            toast.error("Failed to load packs");
        } finally {
            setLoading(false);
        }
    };

    const installPack = async (pack: Pack) => {
        confirm({
            title: "Install Pack",
            message: `Install "${pack.name}"? ${pack.price > 0 ? `This will cost ${pack.price} CR.` : "This pack is free."}`,
            confirmText: "Install",
            onConfirm: async () => {
                try {
                    await api.packs.install(pack.id);
                    setInstalledIds(prev => new Set(prev).add(pack.id));
                    toast.success(`"${pack.name}" installed successfully!`);
                } catch (e) {
                    toast.error("Failed to install pack");
                }
            },
        });
    };

    if (loading) return <div className="p-8 text-center text-muted-foreground">Loading packs...</div>;

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold flex items-center gap-2">
                <Package className="text-purple-400" /> Strategy Packs
            </h1>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {packs.map((pack) => {
                    const isInstalled = installedIds.has(pack.id) || pack.installed;
                    return (
                        <Card key={pack.id} className="glass flex flex-col">
                            <CardHeader>
                                <CardTitle className="flex justify-between items-center">
                                    {pack.name}
                                    <div className="flex gap-2">
                                        {isInstalled && (
                                            <span className="text-xs bg-blue-500/20 text-blue-300 px-2 py-1 rounded flex items-center gap-1">
                                                <Check className="w-3 h-3" /> Installed
                                            </span>
                                        )}
                                        {pack.price === 0 ? (
                                            <span className="text-xs bg-green-500/20 text-green-300 px-2 py-1 rounded">FREE</span>
                                        ) : (
                                            <span className="text-xs bg-yellow-500/20 text-yellow-300 px-2 py-1 rounded">{pack.price} CR</span>
                                        )}
                                    </div>
                                </CardTitle>
                                <CardDescription>{pack.description}</CardDescription>
                            </CardHeader>
                            <CardContent className="flex-1">
                                <div className="space-y-2 text-sm text-muted-foreground">
                                    <div className="flex justify-between">
                                        <span>Episodes</span>
                                        <span className="font-mono text-slate-200">{pack.config.episodes}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Learning Rate</span>
                                        <span className="font-mono text-slate-200">{pack.config.learning_rate}</span>
                                    </div>
                                </div>
                            </CardContent>
                            <CardFooter>
                                <Button
                                    className="w-full"
                                    onClick={() => installPack(pack)}
                                    disabled={isInstalled}
                                    variant={isInstalled ? "outline" : "default"}
                                >
                                    {isInstalled ? (
                                        <><Check className="w-4 h-4 mr-2" /> Installed</>
                                    ) : (
                                        <><Download className="w-4 h-4 mr-2" /> Install Pack</>
                                    )}
                                </Button>
                            </CardFooter>
                        </Card>
                    );
                })}
            </div>
        </div>
    );
}

