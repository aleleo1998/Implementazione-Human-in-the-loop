// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ModelRegistry {
    // Definizione della struttura Model
    struct Model {
        string name;
        string data;
        string id;
        string accuracy;
    }

    // Array dinamico di modelli
    Model[] public models;

    // Funzione per aggiungere un nuovo modello all'array
    function addModel(
        string memory _name,
        string memory _data,
        string memory _id,
        string memory _accuracy
    ) public {
        Model memory newModel = Model(_name, _data, _id, _accuracy);
        models.push(newModel);
    }

    // Funzione per ottenere il numero di modelli registrati
    function getModelsCount() public view returns (uint256) {
        return models.length;
    }

    // Funzione per ottenere i dettagli di un modello per indice nell'array
    function getModel(uint256 index) public view returns (string memory, string memory, string memory, string memory) {
        require(index < models.length, "Index out of bounds");
        Model memory model = models[index];
        return (model.name, model.data, model.id, model.accuracy);
    }

    // Funzione per restituire l'intero array di modelli in un formato leggibile
    function getAllModels() public view returns (string[] memory, string[] memory, string[] memory, string[] memory) {
        uint256 length = models.length;
        string[] memory names = new string[](length);
        string[] memory datas = new string[](length);
        string[] memory ids = new string[](length);
        string[] memory accuracies = new string[](length);

        for (uint256 i = 0; i < length; i++) {
            names[i] = models[i].name;
            datas[i] = models[i].data;
            ids[i] = models[i].id;
            accuracies[i] = models[i].accuracy;
        }

        return (names, datas, ids, accuracies);
    }

    // Funzione per modificare l'accuratezza di un modello
    function updateModelAccuracy(string memory _id, string memory _newAccuracy) public {
        for (uint256 i = 0; i < models.length; i++) {
            if (keccak256(bytes(models[i].id)) == keccak256(bytes(_id))) {
                models[i].accuracy = _newAccuracy;
                return;
            }
        }
    }
}
