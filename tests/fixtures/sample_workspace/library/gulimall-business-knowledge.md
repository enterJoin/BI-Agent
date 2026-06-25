# Gulimall 业务知识库

## 订单与购物车交集

订单系统和购物车系统的交集发生在结算链路。用户在购物车 cart-service
选择商品后进入 checkout，订单服务 order-service 会读取购物车快照、会员地址、
优惠券 coupon、库存 stock 和价格 price，生成 oms_order 主订单和 oms_order_item
明细。关键词：order, orders, order-service, cart, cart-service, checkout,
gulimall_order_route, gulimall_cart_route, oms_order, oms_order_item。

排查“购物车提交订单失败”时优先检查 cart-service 是否传递 skuId、数量、选中状态
和价格快照，再检查 order-service 是否锁定库存、计算优惠券和落库 oms_order。

## 网关路由关键词

商城网关 gateway 通过 gulimall_order_route 把 order.gulimall.com 转发到
order-service，通过 gulimall_cart_route 把 cart.gulimall.com 转发到 cart-service。
当查询“订单系统和购物车系统有什么交集”时，这些路由词应帮助向量检索命中结算链路。
