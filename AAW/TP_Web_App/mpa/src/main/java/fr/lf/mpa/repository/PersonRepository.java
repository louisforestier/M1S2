package fr.lf.mpa.repository;

import fr.lf.mpa.model.Person;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import java.util.UUID;

@Repository
public interface PersonRepository extends CrudRepository<Person, UUID> {

    default Iterable<? extends Person> findByEventId(UUID id){
        System.out.println("toto");
        return findAll();
    }
}
