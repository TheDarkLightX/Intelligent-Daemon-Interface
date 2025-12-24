import React from "react";
import { render, screen } from "@testing-library/react";

jest.mock("next/navigation", () => ({
    useRouter: () => ({
        push: jest.fn(),
    }),
}));

jest.mock("@/lib/api", () => ({
    api: {
        agents: {
            list: jest.fn(),
            delete: jest.fn(),
            load: jest.fn(),
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
import { AgentList } from "@/components/Dashboard/AgentList";

describe("AgentList", () => {
    beforeEach(() => {
        jest.spyOn(console, "error").mockImplementation(() => {});
    });

    afterEach(() => {
        (console.error as jest.Mock).mockRestore?.();
        jest.resetAllMocks();
    });

    it("renders empty state when no agents are returned", async () => {
        (api.agents.list as jest.Mock).mockResolvedValue([]);

        render(<AgentList />);

        await screen.findByText(/no agents found/i);
        expect(api.agents.list).toHaveBeenCalledTimes(1);
    });
});
