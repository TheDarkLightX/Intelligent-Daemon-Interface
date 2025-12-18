"use client";

import React, { useCallback } from 'react';
import ReactFlow, {
    Controls,
    Background,
    useNodesState,
    useEdgesState,
    Edge,
    Node,
    Position,
    MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { cn } from '@/lib/utils';

// Types passed from Wizard logic
type VisualizerProps = {
    inputs: Record<string, boolean>;
    strategy: string;
    layers: number;
    className?: string;
};

export function AgentVisualizer({ inputs, strategy, layers, className }: VisualizerProps) {
    // Dynamically build nodes/edges based on props
    const initialNodes: Node[] = [];
    const initialEdges: Edge[] = [];
    let idCounter = 1;

    // 1. Input Source Nodes
    const selectedInputs = Object.entries(inputs).filter(([_, v]) => v).map(([k]) => k);
    selectedInputs.forEach((input, idx) => {
        initialNodes.push({
            id: `input-${input}`,
            data: { label: input.toUpperCase() },
            position: { x: 50, y: 50 + idx * 80 },
            sourcePosition: Position.Right,
            type: 'input',
            className: 'bg-cyan-900 border-cyan-500 text-cyan-100 rounded-md text-xs font-bold w-32 shadow-[0_0_10px_rgba(6,182,212,0.5)]',
        });
    });

    // 2. Logic Layer Nodes (Visual representation of strategy)
    const strategyNodeId = `strat-${strategy}`;
    initialNodes.push({
        id: strategyNodeId,
        data: { label: `${strategy.replace('_', ' ').toUpperCase()} LOGIC` },
        position: { x: 300, y: 150 },
        targetPosition: Position.Left,
        sourcePosition: Position.Right,
        className: 'bg-violet-900 border-violet-500 text-violet-100 rounded-lg p-4 font-bold w-48 shadow-[0_0_15px_rgba(139,92,246,0.5)]',
    });

    // Edges from inputs to strategy
    selectedInputs.forEach((input) => {
        initialEdges.push({
            id: `e-${input}-strat`,
            source: `input-${input}`,
            target: strategyNodeId,
            animated: true,
            style: { stroke: '#06b6d4' },
        });
    });

    // 3. Processing Layers (if > 1)
    let lastId = strategyNodeId;
    for (let i = 1; i <= layers; i++) {
        const layerId = `layer-${i}`;
        initialNodes.push({
            id: layerId,
            data: { label: `LAYER ${i}` },
            position: { x: 550 + (i - 1) * 150, y: 150 },
            targetPosition: Position.Left,
            sourcePosition: Position.Right,
            type: i === layers ? 'output' : 'default',
            className: 'bg-blue-900 border-blue-500 text-blue-100 rounded-full w-24 h-24 flex items-center justify-center font-bold text-xs',
        });

        initialEdges.push({
            id: `e-${lastId}-${layerId}`,
            source: lastId,
            target: layerId,
            animated: true,
            markerEnd: { type: MarkerType.ArrowClosed, color: '#3b82f6' },
            style: { stroke: '#3b82f6' },
        });
        lastId = layerId;
    }

    // 4. Output Actions
    const actions = ['BUY', 'SELL', 'HOLD'];
    actions.forEach((action, idx) => {
        initialNodes.push({
            id: `action-${action}`,
            data: { label: action },
            position: { x: 550 + layers * 150 + 100, y: 50 + idx * 100 },
            targetPosition: Position.Left,
            type: 'output',
            className: cn(
                'rounded-md px-4 py-2 font-bold text-xs border',
                action === 'BUY' ? 'bg-green-900 border-green-500 text-green-100' :
                    action === 'SELL' ? 'bg-red-900 border-red-500 text-red-100' : 'bg-gray-800 border-gray-500 text-gray-300'
            )
        });
        initialEdges.push({
            id: `e-${lastId}-${action}`,
            source: lastId,
            target: `action-${action}`,
            style: { stroke: '#64748b' },
        });
    });

    const [nodes, , onNodesChange] = useNodesState(initialNodes);
    const [edges, , onEdgesChange] = useEdgesState(initialEdges);

    return (
        <div className={cn("h-[400px] w-full border border-border/50 rounded-lg overflow-hidden glass", className)}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                fitView
                className="bg-slate-950/50"
            >
                <Background color="#333" gap={16} />
                <Controls className="bg-white/10 fill-white" />
            </ReactFlow>
        </div>
    );
}
