package fr.lf.mpa.service;

import fr.lf.mpa.context.Context;
import fr.lf.mpa.form.PersonForm;
import fr.lf.mpa.model.Event;
import fr.lf.mpa.model.EventRecord;
import fr.lf.mpa.model.Person;
import fr.lf.mpa.model.PersonRecord;
import fr.lf.mpa.repository.EventRepository;
import fr.lf.mpa.repository.PersonRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class PersonService {

    @Autowired
    private Context context;

    @Autowired
    private PersonRepository personRepository;

    @Autowired
    private EventRepository eventRepository;

    @Value("false")
    private Boolean withContext;
    public Context getContext() {
        return context;
    }

    public void setContext(Context context) {
        this.context = context;
    }

    public void delPerson(UUID id) {
        if (withContext){
            List<PersonRecord> personsFiltered = context.getPersons()
                    .stream()
                    .filter(p -> !p.getId().equals(id))
                    .collect(Collectors.toList());
            context.updatePersons(personsFiltered);
        } else {
            personRepository.deleteById(id);
        }
    }

    public void addPerson(PersonForm personForm){
        if (withContext) {
            EventRecord eventRecord = context.getEvents().stream().filter(e -> e.getId().equals(personForm.getEventId())).findFirst().get();
            context.getPersons().add(new PersonRecord(UUID.randomUUID(), personForm.getFirstName(), personForm.getLastName(), eventRecord));
        } else {
            Event event = eventRepository.findById(personForm.getEventId()).get();
            Person person = new Person();
            person.setFirstName(personForm.getFirstName())
                    .setLastName(personForm.getLastName())
                    .setEvent(event);
            personRepository.save(person);
        }
    }

    public boolean withContext() {
        return withContext;
    }

    public List<Event> getEvents(){
        List<Event> events = new ArrayList<>();
        eventRepository.findAll().forEach(events::add);
        return events;
    }

    public List<Person> getPersons(){
        List<Person> persons = new ArrayList<>();
        personRepository.findAll().forEach(persons::add);
        return persons;
    }

    public EventRepository getEventRepository() {
        return eventRepository;
    }

    public PersonRepository getPersonRepository() {
        return personRepository;
    }

}
