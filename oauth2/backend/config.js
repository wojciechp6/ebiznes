// config.js
require('dotenv').config();

module.exports = {
  JWT_SECRET: process.env.JWT_SECRET || 'tajnysekret',
  GOOGLE_CLIENT_ID: process.env.GOOGLE_CLIENT_ID || '',
  GOOGLE_CLIENT_SECRET: process.env.GOOGLE_CLIENT_SECRET || '',
  GITHUB_CLIENT_ID: process.env.GITHUB_CLIENT_ID || '',
  GITHUB_CLIENT_SECRET: process.env.GITHUB_CLIENT_SECRET || '',
  BASE_URL: process.env.BASE_URL || 'http://localhost:4000',
};