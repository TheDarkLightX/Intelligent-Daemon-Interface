"use client";

import { createContext, useContext, useState, useCallback, ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// Types
interface ConfirmDialogOptions {
    title: string;
    message: string;
    confirmText?: string;
    cancelText?: string;
    variant?: "default" | "destructive";
    onConfirm: () => void | Promise<void>;
    onCancel?: () => void;
}

interface ConfirmDialogContextValue {
    confirm: (options: ConfirmDialogOptions) => void;
}

const ConfirmDialogContext = createContext<ConfirmDialogContextValue | null>(null);

export function ConfirmDialogProvider({ children }: { children: ReactNode }) {
    const [dialog, setDialog] = useState<ConfirmDialogOptions | null>(null);
    const [loading, setLoading] = useState(false);

    const confirm = useCallback((options: ConfirmDialogOptions) => {
        setDialog(options);
    }, []);

    const handleConfirm = async () => {
        if (!dialog) return;
        setLoading(true);
        try {
            await dialog.onConfirm();
        } finally {
            setLoading(false);
            setDialog(null);
        }
    };

    const handleCancel = () => {
        dialog?.onCancel?.();
        setDialog(null);
    };

    return (
        <ConfirmDialogContext.Provider value={{ confirm }}>
            {children}
            <AnimatePresence>
                {dialog && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 backdrop-blur-sm"
                        onClick={handleCancel}
                    >
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            onClick={(e) => e.stopPropagation()}
                            className={cn(
                                "glass border rounded-lg p-6 max-w-md w-full mx-4 shadow-2xl",
                                dialog.variant === "destructive" ? "border-red-500/30" : "border-border"
                            )}
                        >
                            {/* Header */}
                            <div className="flex items-start gap-4 mb-4">
                                {dialog.variant === "destructive" && (
                                    <div className="p-2 rounded-full bg-red-500/10">
                                        <AlertTriangle className="w-6 h-6 text-red-400" />
                                    </div>
                                )}
                                <div className="flex-1">
                                    <h3 className="font-semibold text-lg">{dialog.title}</h3>
                                    <p className="text-sm text-muted-foreground mt-1">{dialog.message}</p>
                                </div>
                                <button
                                    onClick={handleCancel}
                                    className="p-1 hover:bg-white/10 rounded transition-colors"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>

                            {/* Actions */}
                            <div className="flex justify-end gap-3 mt-6">
                                <Button variant="outline" onClick={handleCancel} disabled={loading}>
                                    {dialog.cancelText || "Cancel"}
                                </Button>
                                <Button
                                    onClick={handleConfirm}
                                    disabled={loading}
                                    className={cn(
                                        dialog.variant === "destructive" && "bg-red-600 hover:bg-red-700"
                                    )}
                                >
                                    {loading ? "..." : dialog.confirmText || "Confirm"}
                                </Button>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </ConfirmDialogContext.Provider>
    );
}

export function useConfirmDialog() {
    const context = useContext(ConfirmDialogContext);
    if (!context) {
        throw new Error("useConfirmDialog must be used within a ConfirmDialogProvider");
    }
    return context;
}
