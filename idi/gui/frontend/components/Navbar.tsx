"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Cpu, Terminal, Settings as SettingsIcon, Trophy, Package } from "lucide-react";

export function Navbar() {
    const pathname = usePathname();

    return (
        <nav className="border-b border-border/40 glass sticky top-0 z-50">
            <div className="container mx-auto flex h-16 items-center px-4">
                <Link href="/" className="flex items-center gap-2 font-bold text-xl mr-8 text-cyan-400">
                    <Cpu className="h-6 w-6" />
                    <span>IDI/IAN</span>
                </Link>
                <div className="flex gap-6">
                    <Link
                        href="/wizard"
                        className={cn(
                            "text-sm font-medium transition-colors hover:text-primary",
                            pathname === "/wizard" ? "text-primary bg-primary/10 px-3 py-1 rounded-full" : "text-muted-foreground"
                        )}
                    >
                        Agent Wizard
                    </Link>
                    <Link
                        href="/training"
                        className={cn(
                            "text-sm font-medium transition-colors hover:text-primary",
                            pathname === "/training" ? "text-primary bg-primary/10 px-3 py-1 rounded-full" : "text-muted-foreground"
                        )}
                    >
                        Training
                    </Link>
                    <Link
                        href="/leaderboard"
                        className={cn(
                            "text-sm font-medium transition-colors hover:text-primary",
                            pathname === "/leaderboard" ? "text-primary bg-primary/10 px-3 py-1 rounded-full" : "text-muted-foreground"
                        )}
                    >
                        <Trophy className="w-4 h-4 inline-block mr-1" /> Leaderboard
                    </Link>
                    <Link
                        href="/packs"
                        className={cn(
                            "text-sm font-medium transition-colors hover:text-primary",
                            pathname === "/packs" ? "text-primary bg-primary/10 px-3 py-1 rounded-full" : "text-muted-foreground"
                        )}
                    >
                        <Package className="w-4 h-4 inline-block mr-1" /> Packs
                    </Link>
                </div>
                <div className="ml-auto flex items-center gap-4">
                    <Link
                        href="/settings"
                        className={cn(
                            "text-sm font-medium transition-colors hover:text-cyan-400",
                            pathname === "/settings" ? "text-cyan-400" : "text-muted-foreground"
                        )}
                    >
                        <SettingsIcon className="w-5 h-5" />
                    </Link>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Terminal className="h-4 w-4" />
                        <span>v1.0.0</span>
                    </div>
                </div>
            </div>
        </nav>
    );
}
