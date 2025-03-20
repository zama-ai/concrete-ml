use tfhe::{ConfigBuilder, generate_keys, FheInt32, FheUint32, set_server_key};
use tfhe::prelude::*;

fn main() {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let random_token: i32 = rng.gen_range(1..=100);
    println!("Random token: {}", random_token);

    let config = ConfigBuilder::default().build();
    let (client_key, server_key) = generate_keys(config);
    set_server_key(server_key);

    let clear_values = vec![0u32, 255u32];
    let encrypted_values = clear_values.iter().copied().map(|v| FheUint32::encrypt(v, &client_key)).collect::<Vec<_>>();

    let diff = &encrypted_values[0] - &encrypted_values[1];
    let decrypted_diff: u32 = diff.decrypt(&client_key);
    println!("Decrypted diff: {}", decrypted_diff);

    let sign = diff.gt(0);
    let decrypted_sign: bool = sign.decrypt(&client_key);
    println!("Decrypted sign: {}", decrypted_sign);

    let int_sign = FheInt32::cast_from(sign);
    let decrypted_int_sign: i32 = int_sign.decrypt(&client_key);
    println!("Decrypted int sign: {}", decrypted_int_sign);

    let response = int_sign * random_token;
    let decrypted_response: i32 = response.decrypt(&client_key);
    println!("Final response: {}", decrypted_response);

}

