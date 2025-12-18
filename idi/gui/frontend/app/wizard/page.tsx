import { Wizard } from "@/components/Wizard/Wizard";

export default function WizardPage() {
    return (
        <div className="flex flex-col items-center justify-center min-h-[80vh]">
            <div className="w-full max-w-4xl">
                <h2 className="text-3xl font-bold mb-8 text-center bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent">
                    Agent Factory Wizard
                </h2>
                <Wizard />
            </div>
        </div>
    );
}
