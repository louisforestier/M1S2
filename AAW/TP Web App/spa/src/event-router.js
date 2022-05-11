const express = require("express");
const {v4} = require("uuid");
const router = express.Router();

let events = [
    {id:v4(),name:'Paris Manga',date:'2022-06-09'},
    {id:v4(),name:'Japan Expo',date:'2022-11-11'}]

router.get("/",(req,res) => {
    res.send(events);
})
router.post("/",(req,res) => {
    const event = req.body;
    console.log(event);
    event.id = v4();
    events.push(event);
    res.send(events);
})
router.delete("/:id",(req,res) => {
    events = events.filter(value => value.id != req.params.id)
    res.send(events);
})

module.exports= {
    eventRouter:router,
    events
}