const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors({ origin: process.env.FRONTEND_URL || 'http://localhost:3000' }));
app.use(express.json());

app.use('/api/games',       require('./routes/games'));
app.use('/api/predictions', require('./routes/predictions'));
app.use('/api/teams',       require('./routes/teams'));

app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.listen(PORT, () => console.log(`API running on http://localhost:${PORT}`));