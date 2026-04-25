// frontend/next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  turbopack: {
    // Explicitly set the root to the frontend directory
    root: __dirname,
  },
};

module.exports = nextConfig;