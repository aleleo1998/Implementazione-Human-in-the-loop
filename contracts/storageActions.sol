// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StorageAction {
    struct Action {
        string user;
        string data;
        string description;
    }

    Action[] private actions;

    // Evento per notificare una nuova azione
    event ActionStored(string user, string data, string description);

    // Metodo per memorizzare una nuova azione
    function storeAction(string memory _user, string memory _data, string memory _description) public {
        actions.push(Action(_user, _data, _description));
        emit ActionStored(_user, _data, _description);
    }

    // Metodo per ottenere tutte le azioni memorizzate in un formato leggibile
    function getActions() public view returns (string[] memory, string[] memory, string[] memory) {
        uint256 length = actions.length;
        string[] memory users = new string[](length);
        string[] memory datas = new string[](length);
        string[] memory descriptions = new string[](length);

        for (uint256 i = 0; i < length; i++) {
            users[i] = actions[i].user;
            datas[i] = actions[i].data;
            descriptions[i] = actions[i].description;
        }

        return (users, datas, descriptions);
    }
}