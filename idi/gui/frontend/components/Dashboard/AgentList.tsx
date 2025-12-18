"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Bot, RefreshCw, Trash2, Edit } from "lucide-react";
import { useRouter } from "next/navigation";

type Agent = {
    name: string;
    strategy: string;
    created_at: number;
    path: string;
};

export function AgentList() {
    const [agents, setAgents] = useState<Agent[]>([]);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    useEffect(() => {
        loadAgents();
    }, []);

    const loadAgents = async () => {
        try {
            setLoading(true);
            const data = await api.agents.list();
            setAgents(data);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (name: string) => {
        if (!confirm(`Are you sure you want to delete ${name}?`)) return;
        try {
            await api.agents.delete(name);
            loadAgents();
        } catch (e) {
            console.error(e);
        }
    };

    const handleLoad = async (name: string) => {
        try {
            await api.agents.load(name);
            router.push("/wizard");
        } catch (e) {
            console.error(e);
            alert("Failed to load agent");
        }
    };

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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {agents.map((agent) => (
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
    );
}
