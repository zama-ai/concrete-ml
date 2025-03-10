use std::path::Path;
use std::fs;
use std::io::Cursor;
use bincode;
use tfhe::core_crypto::prelude::LweSecretKey;
use tfhe::set_server_key;
use tfhe::safe_serialization::safe_deserialize;
use tfhe::FheUint8;
use std::error::Error;
use tfhe::ServerKey;
use tfhe::prelude::*;
use tfhe::{FheInt32, FheUint32};

const SERIALIZE_SIZE_LIMIT: u64 = 900_000_000_000_000;


fn _load_lwe_sk(path: &String) -> Result<LweSecretKey<Vec<u64>>, Box<dyn Error>> {
    let file = fs::File::open(path)?;
    let data = safe_deserialize(file, SERIALIZE_SIZE_LIMIT)?;
    Ok(data)
}

fn load_pk(path: &String) -> Result<ServerKey, Box<dyn Error>>  {
    let file = fs::File::open(path)?;
    let data: ServerKey = safe_deserialize(file, SERIALIZE_SIZE_LIMIT)?;
    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let pk_key_path = String::from("serverKey.bin");
    let pk_key = load_pk(&pk_key_path)?;
    set_server_key(pk_key);
    let encrypted_data_path = String::from("encrypted_input.bin");

    let path: &Path = Path::new(&encrypted_data_path);
    let serialized_list = fs::read(path).unwrap();
    let mut serialized_data = Cursor::new(serialized_list);

    let encrypted_values: Vec<FheUint8> = bincode::deserialize_from(&mut serialized_data).unwrap();

    use rand::Rng;

    let mut rng = rand::thread_rng();
    let random_token: i64 = rng.gen_range(1..=100);
    println!("Random token: {}", random_token);


    let diff = &encrypted_values[0] - &encrypted_values[1];

    let _sign = diff.gt(0);
    // let int_sign = FheInt32::cast_from(sign);
    // let response = int_sign * random_token;

    Ok(())
}
