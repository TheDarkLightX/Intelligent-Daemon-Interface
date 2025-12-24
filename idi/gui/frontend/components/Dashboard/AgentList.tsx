"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Bot, RefreshCw, Trash2, Edit, Search } from "lucide-react";
import { useRouter } from "next/navigation";
import { useToast } from "@/components/ui/toast";
import { useConfirmDialog } from "@/components/ui/confirm-dialog";

type Agent = {
    name: string;
    strategy: string;
    created_at: number;
    path: string;
};

export function AgentList() {
    const [agents, setAgents] = useState<Agent[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState("");
    const router = useRouter();
    const toast = useToast();
    const { confirm } = useConfirmDialog();

    useEffect(() => {
        loadAgents();
    }, []);

    const loadAgents = async () => {
        try {
            setLoading(true);
            const data = await api.agents.list() as Agent[];
            setAgents(data);
        } catch (e) {
            console.error(e);
            toast.error("Failed to load agents");
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (name: string) => {
        confirm({
            title: "Delete Agent",
            message: `Are you sure you want to delete "${name}"? This action cannot be undone.`,
            confirmText: "Delete",
            variant: "destructive",
            onConfirm: async () => {
                try {
                    await api.agents.delete(name);
                    toast.success(`Agent "${name}" deleted successfully`);
                    loadAgents();
                } catch (e) {
                    console.error(e);
                    toast.error("Failed to delete agent");
                }
            },
        });
    };

    const handleLoad = async (name: string) => {
        try {
            await api.agents.load(name);
            router.push("/wizard");
        } catch (e) {
            console.error(e);
            toast.error("Failed to load agent");
        }
    };

    // Filter agents by search query
    const filteredAgents = agents.filter((agent) =>
        agent.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        agent.strategy.toLowerCase().includes(searchQuery.toLowerCase())
    );

    if (loading) return <div className="text-center p-8">Loading Agents...</div>;

    if (agents.length === 0) {
        return (
            <div className="text-center p-8 border border-dashed rounded-lg text-muted-foreground glass">
                <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No agents found. Create one in the Wizard!</p>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            {/* Search Bar */}
            <div className="relative max-w-sm">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                    type="text"
                    placeholder="Search agents..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 rounded-md bg-secondary/50 border border-input text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                />
            </div>

            {/* Agent Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredAgents.map((agent) => (
                    <Card key={agent.name} className="glass hover:border-primary/50 transition-colors group">
                        <CardHeader>
                            <CardTitle className="flex justify-between items-start">
                                <span>{agent.name}</span>
                                <Bot className="w-5 h-5 text-muted-foreground group-hover:text-cyan-400 transition-colors" />
                            </CardTitle>
                            <CardDescription>
                                Strategy: <span className="font-mono text-xs bg-secondary px-1 rounded capitalize">{agent.strategy}</span>
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="text-xs text-muted-foreground">
                                Created: {new Date(agent.created_at * 1000).toLocaleDateString()}
                            </div>
                        </CardContent>
                        <CardFooter className="flex justify-between gap-2">
                            <Button
                                variant="outline"
                                size="sm"
                                className="flex-1"
                                onClick={() => handleLoad(agent.name)}
                            >
                                <Edit className="w-4 h-4 mr-2" /> Edit
                            </Button>
                            <Button
                                variant="destructive"
                                size="sm"
                                onClick={() => handleDelete(agent.name)}
                            >
                                <Trash2 className="w-4 h-4" />
                            </Button>
                        </CardFooter>
                    </Card>
                ))}
            </div>

            {/* No results state */}
            {filteredAgents.length === 0 && searchQuery && (
                <div className="text-center p-8 text-muted-foreground">
                    No agents matching "{searchQuery}"
                </div>
            )}
        </div>
    );
}

