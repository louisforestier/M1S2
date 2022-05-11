const express = require("express");
const router = express.Router();

const {eventRouter} = require("./event-router");
const {personRouter} = require("./person-router");

router.use("/events",eventRouter);
router.use("/persons",personRouter);

module.exports =router;