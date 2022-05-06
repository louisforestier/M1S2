package fr.lf.mpa.controller;

import fr.lf.mpa.form.PersonForm;
import fr.lf.mpa.model.EventRecord;
import fr.lf.mpa.model.PersonRecord;
import fr.lf.mpa.service.PersonService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;
import java.util.stream.Collectors;

@Controller
public class PersonController extends BaseController{

    @Autowired
    private PersonService personService;

    @GetMapping(value = {"/persons"})
    public String showPersonListPage(Model model) {
        initModel(model);
        model.addAttribute("persons",personService.getContext().getPersons());
        return "person_list";
    }

    @GetMapping(value = {"/persons/add"})
    public String showAddPersonPage(Model model, @RequestParam(required = false) UUID event) {
        initModel(model);
        PersonForm personForm = new PersonForm();
        model.addAttribute("events",personService.getContext().getEvents());
        model.addAttribute("personForm",personForm);
        return "add_person";
    }

    @PostMapping(value = {"/persons"})
    public String savePerson(Model model, @ModelAttribute("personForm") PersonForm personForm) {
        initModel(model);
        EventRecord eventRecord = personService.getContext().getEvents().stream().filter(p -> p.getId().equals(personForm.getEventId())).findFirst().get();
        personService.getContext().getPersons().add(new PersonRecord(UUID.randomUUID(), personForm.getFirstName(), personForm.getLastName(),eventRecord));
        return "redirect:/persons";
    }

    @DeleteMapping(value = {"/persons/{id}"})
    public String delPerson(Model model, @PathVariable("id") UUID id) {
        if (id != null) {
            personService.delPerson(id);
            return "redirect:/persons";
        }
        return "index";
    }
}
