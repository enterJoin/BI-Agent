package com.example.demo.repository;

import com.example.demo.model.User;
import org.springframework.stereotype.Repository;

@Repository
public class UserRepository {
    public User findById(Long id) {
        return new User(id, "admin");
    }
}

