package com.example.order.entity;

import com.baomidou.mybatisplus.annotation.TableName;

@TableName("oms_order")
public class OrderEntity {
    private Long id;
    private String orderSn;
    private Integer status;
}
