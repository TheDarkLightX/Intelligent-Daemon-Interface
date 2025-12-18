import React from 'react';
import { cn } from "@/lib/utils";

interface CodeBlockProps {
    code: string;
    language?: string;
    showLineNumbers?: boolean;
    className?: string;
}

export const CodeBlock = ({ code, language = 'text', showLineNumbers = false, className }: CodeBlockProps) => {
    return (
        <pre className={cn("p-4 rounded-lg bg-black/50 overflow-x-auto font-mono text-sm border border-border", className)}>
            <code className={`language-${language}`}>
                {code}
            </code>
        </pre>
    );
};
