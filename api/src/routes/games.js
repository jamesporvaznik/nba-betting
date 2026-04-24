const express = require('express');
const router = express.Router();

router.get('/', async (req, res) => {
  res.json({ data: [], message: 'DB not connected yet' });
});

router.get('/:id', async (req, res) => {
  res.json({ id: req.params.id, message: 'DB not connected yet' });
});

module.exports = router;