package fr.lf.mpa.service;

import fr.lf.mpa.context.Context;
import fr.lf.mpa.model.EventRecord;
import fr.lf.mpa.model.PersonRecord;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class PersonService {

    @Autowired
    private Context context;

    @Value("true")
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
        }
    }
}
