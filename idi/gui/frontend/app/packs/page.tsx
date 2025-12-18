"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Navbar } from "@/components/Navbar";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Package, Download, Check } from "lucide-react";

export default function PacksPage() {
    const [packs, setPacks] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        api.packs.list().then(setPacks).finally(() => setLoading(false));
    }, []);

    const installPack = async (id: string) => {
        try {
            await api.packs.install(id);
            alert("Pack installed! Check 'My Agents' or Wizard.");
        } catch (e) {
            alert("Failed to install pack");
        }
    };

    return (
        <main className="min-h-screen bg-slate-950 text-slate-100 pb-20">
            <Navbar />
            <div className="container mx-auto p-8">
                <h1 className="text-3xl font-bold mb-6 flex items-center gap-2">
                    <Package className="text-purple-400" /> Strategy Packs
                </h1>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {packs.map((pack) => (
                        <Card key={pack.id} className="glass flex flex-col">
                            <CardHeader>
                                <CardTitle className="flex justify-between items-center">
                                    {pack.name}
                                    {pack.price === 0 ? (
                                        <span className="text-xs bg-green-500/20 text-green-300 px-2 py-1 rounded">FREE</span>
                                    ) : (
                                        <span className="text-xs bg-yellow-500/20 text-yellow-300 px-2 py-1 rounded">{pack.price} CR</span>
                                    )}
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
                                <Button className="w-full" onClick={() => installPack(pack.id)}>
                                    <Download className="w-4 h-4 mr-2" /> Install Pack
                                </Button>
                            </CardFooter>
                        </Card>
                    ))}
                </div>
            </div>
        </main>
    );
}
