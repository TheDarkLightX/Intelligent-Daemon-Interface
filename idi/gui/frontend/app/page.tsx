"use client";
import { motion } from "framer-motion";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Bot, LineChart } from "lucide-react";

import { AgentList } from "@/components/Dashboard/AgentList";

export default function Home() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="flex flex-col gap-12 py-8"
    >
      <motion.div variants={item} className="text-center space-y-4">
        <h1 className="text-4xl md:text-6xl font-bold tracking-tighter bg-gradient-to-r from-white to-gray-500 bg-clip-text text-transparent">
          Intelligent Daemon Interface
        </h1>
        <p className="text-xl text-muted-foreground max-w-[600px] mx-auto">
          Design, train, and deploy advanced Tau Agents with the power of IAN.
        </p>
      </motion.div>

      <motion.div variants={item} className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl mx-auto">
        <Link href="/wizard">
          <Card className="h-full hover:border-cyan-500/50 transition-colors cursor-pointer group glass">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bot className="h-6 w-6 text-cyan-400" />
                Agent Wizard
              </CardTitle>
              <CardDescription>
                Create new agents step-by-step using the Tau Factory.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center text-sm text-cyan-400 group-hover:translate-x-1 transition-transform">
                Start Creation <ArrowRight className="ml-2 h-4 w-4" />
              </div>
            </CardContent>
          </Card>
        </Link>

        <Link href="/training">
          <Card className="h-full hover:border-violet-500/50 transition-colors cursor-pointer group glass">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <LineChart className="h-6 w-6 text-violet-400" />
                IAN Trainer
              </CardTitle>
              <CardDescription>
                Train Q-tables with real-time feedback and visualization.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center text-sm text-violet-400 group-hover:translate-x-1 transition-transform">
                Open Monitor <ArrowRight className="ml-2 h-4 w-4" />
              </div>
            </CardContent>
          </Card>
        </Link>
      </motion.div>

      <motion.div variants={item} className="space-y-6">
        <div className="flex items-center justify-between border-b border-border pb-2">
          <h2 className="text-2xl font-bold tracking-tight">My Agents</h2>
        </div>
        <AgentList />
      </motion.div>
    </motion.div>
  );
}
