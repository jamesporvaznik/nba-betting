const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');

router.get('/', async (req, res) => {
  res.json({ data: [], message: 'DB not connected yet' });
});

router.post('/run/:gameId', (req, res) => {
  const pythonDir = path.join(__dirname, '../../../python');
  const py = spawn('python3', ['models/predict.py', req.params.gameId], {
    cwd: pythonDir
  });
  let result = '', errorOut = '';
  py.stdout.on('data', d => result += d.toString());
  py.stderr.on('data', d => errorOut += d.toString());
  py.on('close', code => {
    if (code !== 0) return res.status(500).json({ error: errorOut });
    try { res.json(JSON.parse(result)); }
    catch { res.status(500).json({ error: 'Invalid output', raw: result }); }
  });
});

module.exports = router;