import { ApiError, api } from "@/lib/api";

describe("api", () => {
    const originalFetch = global.fetch;

    afterEach(() => {
        global.fetch = originalFetch;
        jest.resetAllMocks();
    });

    it("throws ApiError with structured detail when response is not ok", async () => {
        global.fetch = jest.fn(async () =>
            Promise.resolve({
                ok: false,
                status: 400,
                statusText: "Bad Request",
                json: async () => ({ detail: "invalid" }),
            } as any),
        );

        await expect(api.wizard.getState()).rejects.toEqual(
            expect.objectContaining({
                name: "ApiError",
                status: 400,
                detail: "invalid",
            }),
        );
    });

    it("falls back to statusText when error response body is not json", async () => {
        global.fetch = jest.fn(async () =>
            Promise.resolve({
                ok: false,
                status: 500,
                statusText: "Server Error",
                json: async () => {
                    throw new Error("not json");
                },
            } as any),
        );

        await expect(api.wizard.getState()).rejects.toEqual(
            expect.objectContaining({
                status: 500,
                detail: "Server Error",
            }),
        );
    });

    it("returns parsed json when response is ok", async () => {
        global.fetch = jest.fn(async () =>
            Promise.resolve({
                ok: true,
                json: async () => ({ ok: true }),
            } as any),
        );

        await expect(api.wizard.getState()).resolves.toEqual({ ok: true });
    });

    it("ApiError is an Error subtype with status and detail", () => {
        const err = new ApiError(418, "teapot");
        expect(err).toBeInstanceOf(Error);
        expect(err.status).toBe(418);
        expect(err.detail).toBe("teapot");
    });
});
