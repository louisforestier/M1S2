package fr.lf.mpa.controller;

import fr.lf.mpa.form.EventForm;
import fr.lf.mpa.service.EventService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@Controller
public class EventController extends BaseController{

    @Autowired
    private EventService eventService;

    @GetMapping(value = {"/events"})
    public String showEventListPage(Model model) {
        initModel(model);
        if (eventService.withContext())
            model.addAttribute("events",eventService.getContext().getEvents());
        else model.addAttribute("events",eventService.getEvents());
        return "event_list";
    }

    @GetMapping(value = {"/events/add"})
    public String showAddEventPage(Model model) {
        initModel(model);
        EventForm eventForm = new EventForm();
        model.addAttribute("eventForm",eventForm);
        return "add_event";
    }

    @PostMapping(value = {"/events"})
    public String saveEvent(Model model, @ModelAttribute("eventForm") EventForm eventForm) {
        initModel(model);
        eventService.addEvent(eventForm);
        return "redirect:/events";
    }

    @DeleteMapping(value = {"/events/{id}"})
    public String delEvent(Model model, @PathVariable("id") UUID id) {
        if (id != null) {
            eventService.delEvent(id);
            return "redirect:/events";
        }
        return "index";
    }
}
