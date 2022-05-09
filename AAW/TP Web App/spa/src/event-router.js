const express = require("express");
const router = express.Router();

let events = [
    {name:'Paris Manga',date:'09/06/2022'},
    {name:'Japan Expo',date:'11/11/2022'}]

router.get("/",(req,res) => {
    res.send(events);
})
router.post("/",(req,res) => {
    res.send("TODO");
})
router.delete("/:id",(req,res) => {
    res.send("TODO");
})

module.exports= {
    eventRouter:router,
    events
}