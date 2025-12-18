import { TrainingMonitor } from "@/components/Training/TrainingMonitor";

export default function TrainingPage() {
    return (
        <div className="flex flex-col items-center min-h-[80vh] py-8">
            <div className="w-full max-w-6xl">
                <h2 className="text-3xl font-bold mb-8 text-center bg-gradient-to-r from-violet-400 to-purple-600 bg-clip-text text-transparent">
                    IAN Training Monitor
                </h2>
                <TrainingMonitor />
            </div>
        </div>
    );
}
