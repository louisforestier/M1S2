package fr.lf.mpa.repository;

import fr.lf.mpa.model.Event;
import fr.lf.mpa.model.Person;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import java.util.UUID;

@Repository
public interface EventRepository extends CrudRepository<Event, UUID> {
}
