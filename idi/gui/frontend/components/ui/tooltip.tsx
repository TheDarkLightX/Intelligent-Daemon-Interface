"use client";

import { createContext, useContext, useState, ReactNode } from "react";
import { Info } from "lucide-react";
import { cn } from "@/lib/utils";

interface TooltipProps {
    content: string;
    children: ReactNode;
    side?: "top" | "bottom" | "left" | "right";
}

export function Tooltip({ content, children, side = "top" }: TooltipProps) {
    const [isVisible, setIsVisible] = useState(false);

    const positionClasses = {
        top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
        bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
        left: "right-full top-1/2 -translate-y-1/2 mr-2",
        right: "left-full top-1/2 -translate-y-1/2 ml-2",
    };

    return (
        <div
            className="relative inline-block"
            onMouseEnter={() => setIsVisible(true)}
            onMouseLeave={() => setIsVisible(false)}
        >
            {children}
            {isVisible && (
                <div
                    className={cn(
                        "absolute z-50 px-3 py-2 text-xs rounded-md bg-popover border border-border shadow-lg max-w-[200px] pointer-events-none",
                        positionClasses[side]
                    )}
                >
                    {content}
                </div>
            )}
        </div>
    );
}

// Info tooltip icon variant
export function InfoTooltip({ content }: { content: string }) {
    return (
        <Tooltip content={content} side="top">
            <Info className="w-4 h-4 text-muted-foreground hover:text-cyan-400 cursor-help transition-colors" />
        </Tooltip>
    );
}
