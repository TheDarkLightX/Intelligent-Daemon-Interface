import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

jest.mock("framer-motion", () => ({
    motion: {
        div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    },
    AnimatePresence: ({ children }: any) => <div>{children}</div>,
}));

jest.mock("@/components/Wizard/AgentVisualizer", () => ({
    AgentVisualizer: () => <div data-testid="agent-visualizer" />,
}));

jest.mock("@/lib/api", () => ({
    api: {
        wizard: {
            getState: jest.fn(),
            next: jest.fn(),
            prev: jest.fn(),
            reset: jest.fn(),
            getSpec: jest.fn(),
            export: jest.fn(),
        },
        agents: {
            save: jest.fn(),
        },
    },
}));

jest.mock("@/components/ui/toast", () => ({
    ToastProvider: ({ children }: any) => <>{children}</>,
    useToast: () => ({
        toast: jest.fn(),
        success: jest.fn(),
        error: jest.fn(),
        info: jest.fn(),
        warning: jest.fn(),
    }),
}));

jest.mock("@/components/ui/confirm-dialog", () => ({
    ConfirmDialogProvider: ({ children }: any) => <>{children}</>,
    useConfirmDialog: () => ({
        confirm: jest.fn(),
    }),
}));

import { api } from "@/lib/api";
import { Wizard } from "@/components/Wizard/Wizard";

describe("Wizard", () => {
    beforeEach(() => {
        jest.spyOn(console, "error").mockImplementation(() => {});
    });

    afterEach(() => {
        (console.error as jest.Mock).mockRestore?.();
        jest.resetAllMocks();
    });

    it("renders an API error banner when next step fails", async () => {
        (api.wizard.getState as jest.Mock).mockResolvedValue({
            current_step_idx: 0,
            data: {
                name: "agent",
                strategy: "momentum",
                selected_inputs: {},
                num_layers: 1,
                include_safety: false,
                include_communication: false,
            },
            validation_errors: {},
        });

        (api.wizard.next as jest.Mock).mockRejectedValue({ detail: "nope" });

        render(<Wizard />);

        const nextButton = await screen.findByRole("button", { name: /next/i });
        await waitFor(() => expect(nextButton).not.toBeDisabled());

        fireEvent.click(nextButton);

        await screen.findByText("nope");
        expect(api.wizard.next).toHaveBeenCalledTimes(1);
    });
});
