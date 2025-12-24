const nextJest = require("next/jest");

const createJestConfig = nextJest({
    dir: "./",
});

/** @type {import('jest').Config} */
const customJestConfig = {
    testEnvironment: "jsdom",
    setupFilesAfterEnv: ["<rootDir>/jest.setup.js"],
    moduleNameMapper: {
        "^@/(.*)$": "<rootDir>/$1",
        "@next/font/(.*)": "<rootDir>/__mocks__/nextFontMock.js",
        "next/font/(.*)": "<rootDir>/__mocks__/nextFontMock.js",
        "^.+\\.(css|less|scss|sass)$": "<rootDir>/__mocks__/styleMock.js",
        "^.+\\.(png|jpg|jpeg|gif|webp|avif|ico|bmp|svg)$": "<rootDir>/__mocks__/fileMock.js",
    },
    testPathIgnorePatterns: ["<rootDir>/.next/", "<rootDir>/node_modules/"],
};

module.exports = createJestConfig(customJestConfig);
