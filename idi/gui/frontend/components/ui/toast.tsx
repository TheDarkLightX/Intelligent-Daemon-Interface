"use client";

import { createContext, useContext, useState, useCallback, ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle, XCircle, Info, AlertTriangle, X } from "lucide-react";
import { cn } from "@/lib/utils";

// Types
type ToastType = "success" | "error" | "info" | "warning";

interface Toast {
    id: string;
    type: ToastType;
    message: string;
    duration?: number;
}

interface ToastContextValue {
    toast: (type: ToastType, message: string, duration?: number) => void;
    success: (message: string, duration?: number) => void;
    error: (message: string, duration?: number) => void;
    info: (message: string, duration?: number) => void;
    warning: (message: string, duration?: number) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const TOAST_ICONS = {
    success: CheckCircle,
    error: XCircle,
    info: Info,
    warning: AlertTriangle,
};

const TOAST_STYLES = {
    success: "border-green-500/50 bg-green-500/10 text-green-400",
    error: "border-red-500/50 bg-red-500/10 text-red-400",
    info: "border-cyan-500/50 bg-cyan-500/10 text-cyan-400",
    warning: "border-yellow-500/50 bg-yellow-500/10 text-yellow-400",
};

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: string) => void }) {
    const Icon = TOAST_ICONS[toast.type];

    return (
        <motion.div
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            className={cn(
                "flex items-center gap-3 px-4 py-3 rounded-lg border backdrop-blur-sm shadow-lg min-w-[300px] max-w-[400px]",
                TOAST_STYLES[toast.type]
            )}
        >
            <Icon className="w-5 h-5 flex-shrink-0" />
            <span className="text-sm flex-1">{toast.message}</span>
            <button
                onClick={() => onDismiss(toast.id)}
                className="p-1 hover:bg-white/10 rounded transition-colors"
            >
                <X className="w-4 h-4" />
            </button>
        </motion.div>
    );
}

export function ToastProvider({ children }: { children: ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const dismissToast = useCallback((id: string) => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
    }, []);

    const addToast = useCallback((type: ToastType, message: string, duration = 4000) => {
        const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
        const newToast: Toast = { id, type, message, duration };

        setToasts((prev) => [...prev, newToast]);

        if (duration > 0) {
            setTimeout(() => dismissToast(id), duration);
        }
    }, [dismissToast]);

    const contextValue: ToastContextValue = {
        toast: addToast,
        success: (msg, dur) => addToast("success", msg, dur),
        error: (msg, dur) => addToast("error", msg, dur),
        info: (msg, dur) => addToast("info", msg, dur),
        warning: (msg, dur) => addToast("warning", msg, dur),
    };

    return (
        <ToastContext.Provider value={contextValue}>
            {children}
            {/* Toast Container - fixed to top-right */}
            <div className="fixed top-20 right-4 z-[100] flex flex-col gap-2 pointer-events-none">
                <AnimatePresence>
                    {toasts.map((toast) => (
                        <div key={toast.id} className="pointer-events-auto">
                            <ToastItem toast={toast} onDismiss={dismissToast} />
                        </div>
                    ))}
                </AnimatePresence>
            </div>
        </ToastContext.Provider>
    );
}

export function useToast() {
    const context = useContext(ToastContext);
    if (!context) {
        throw new Error("useToast must be used within a ToastProvider");
    }
    return context;
}
